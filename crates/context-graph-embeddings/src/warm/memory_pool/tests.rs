//! Tests for memory pool management.

use std::time::Instant;

use super::pools::WarmMemoryPools;
use crate::warm::config::WarmConfig;
use crate::warm::error::WarmError;

/// One gigabyte in bytes (used in tests).
const GB: usize = 1024 * 1024 * 1024;

#[test]
fn test_rtx_5090_factory_capacities() {
    let pools = WarmMemoryPools::rtx_5090();

    // 24GB for model pool
    assert_eq!(pools.model_pool_capacity(), 24 * GB);
    // 8GB for working pool
    assert_eq!(pools.working_pool_capacity(), 8 * GB);
    // 32GB total
    assert_eq!(pools.total_capacity(), 32 * GB);
}

#[test]
fn test_model_allocation_and_deallocation() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Allocate a model
    let result = pools.allocate_model("E1_Semantic", 800_000_000, 0x1000);
    assert!(result.is_ok());

    // Check allocation tracking
    assert!(pools.get_model_allocation("E1_Semantic").is_some());
    let alloc = pools.get_model_allocation("E1_Semantic").unwrap();
    assert_eq!(alloc.size_bytes, 800_000_000);
    assert_eq!(alloc.vram_ptr, 0x1000);

    // Verify allocated bytes
    assert_eq!(pools.total_allocated_bytes(), 800_000_000);
    assert_eq!(pools.available_model_bytes(), 24 * GB - 800_000_000);

    // Free the model
    let result = pools.free_model("E1_Semantic");
    assert!(result.is_ok());

    // Verify freed
    assert!(pools.get_model_allocation("E1_Semantic").is_none());
    assert_eq!(pools.total_allocated_bytes(), 0);
}

#[test]
fn test_budget_enforcement_model_pool() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Fill most of the model pool
    pools.allocate_model("model1", 20 * GB, 0x1000).unwrap();

    // This should succeed (still within budget)
    let result = pools.allocate_model("model2", 3 * GB, 0x2000);
    assert!(result.is_ok());

    // This should fail (exceeds capacity)
    let result = pools.allocate_model("model3", 2 * GB, 0x3000);
    assert!(result.is_err());
    match result {
        Err(WarmError::VramAllocationFailed { .. }) => (),
        _ => panic!("Expected VramAllocationFailed error"),
    }
}

#[test]
fn test_working_memory_allocation() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Allocate working memory
    let result = pools.allocate_working(GB);
    assert!(result.is_ok());
    assert_eq!(pools.available_working_bytes(), 7 * GB);

    // Free working memory
    pools.free_working(GB).unwrap();
    assert_eq!(pools.available_working_bytes(), 8 * GB);
}

#[test]
fn test_working_memory_exhaustion() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Fill the working pool
    pools.allocate_working(8 * GB).unwrap();

    // This should fail
    let result = pools.allocate_working(1);
    assert!(result.is_err());
    match result {
        Err(WarmError::WorkingMemoryExhausted { .. }) => (),
        _ => panic!("Expected WorkingMemoryExhausted error"),
    }
}

#[test]
fn test_is_within_budget() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Initially within budget
    assert!(pools.is_within_budget());

    // Allocate within limits
    pools.allocate_model("model1", 20 * GB, 0x1000).unwrap();
    pools.allocate_working(6 * GB).unwrap();
    assert!(pools.is_within_budget());

    // Fill exactly to capacity - still within budget
    pools.allocate_model("model2", 4 * GB, 0x2000).unwrap();
    pools.allocate_working(2 * GB).unwrap();
    assert!(pools.is_within_budget());
}

#[test]
fn test_duplicate_model_allocation() {
    let mut pools = WarmMemoryPools::rtx_5090();

    pools.allocate_model("E1_Semantic", GB, 0x1000).unwrap();

    // Duplicate should fail
    let result = pools.allocate_model("E1_Semantic", GB, 0x2000);
    assert!(result.is_err());
    match result {
        Err(WarmError::ModelAlreadyRegistered { model_id }) => {
            assert_eq!(model_id, "E1_Semantic");
        }
        _ => panic!("Expected ModelAlreadyRegistered error"),
    }
}

#[test]
fn test_free_nonexistent_model() {
    let mut pools = WarmMemoryPools::rtx_5090();

    let result = pools.free_model("nonexistent");
    assert!(result.is_err());
    match result {
        Err(WarmError::ModelNotRegistered { model_id }) => {
            assert_eq!(model_id, "nonexistent");
        }
        _ => panic!("Expected ModelNotRegistered error"),
    }
}

#[test]
fn test_list_model_allocations() {
    let mut pools = WarmMemoryPools::rtx_5090();

    pools.allocate_model("model1", GB, 0x1000).unwrap();
    pools.allocate_model("model2", 2 * GB, 0x2000).unwrap();
    pools.allocate_model("model3", 3 * GB, 0x3000).unwrap();

    let allocations = pools.list_model_allocations();
    assert_eq!(allocations.len(), 3);

    // Verify the allocations are tracked
    let ids: Vec<_> = allocations.iter().map(|a| a.model_id.as_str()).collect();
    assert!(ids.contains(&"model1"));
    assert!(ids.contains(&"model2"));
    assert!(ids.contains(&"model3"));
}

#[test]
fn test_model_allocation_timestamp() {
    let mut pools = WarmMemoryPools::rtx_5090();
    let before = Instant::now();

    pools.allocate_model("model1", GB, 0x1000).unwrap();

    let alloc = pools.get_model_allocation("model1").unwrap();
    assert!(alloc.allocated_at >= before);
    assert!(alloc.age_secs() < 1.0); // Should be very recent
}

#[test]
fn test_reset_working_pool() {
    let mut pools = WarmMemoryPools::rtx_5090();

    pools.allocate_working(5 * GB).unwrap();
    assert_eq!(pools.available_working_bytes(), 3 * GB);

    pools.reset_working_pool();
    assert_eq!(pools.available_working_bytes(), 8 * GB);
}

#[test]
fn test_utilization_metrics() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Initially 0% utilization
    assert_eq!(pools.model_pool_utilization(), 0.0);
    assert_eq!(pools.working_pool_utilization(), 0.0);

    // Allocate half of each pool
    pools.allocate_model("model1", 12 * GB, 0x1000).unwrap();
    pools.allocate_working(4 * GB).unwrap();

    // Should be 50% utilization
    assert!((pools.model_pool_utilization() - 0.5).abs() < 0.001);
    assert!((pools.working_pool_utilization() - 0.5).abs() < 0.001);
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_custom_config() {
    let mut config = WarmConfig::default();
    config.vram_budget_bytes = 16 * GB;
    config.vram_headroom_bytes = 4 * GB;

    let pools = WarmMemoryPools::new(config);

    assert_eq!(pools.model_pool_capacity(), 16 * GB);
    assert_eq!(pools.working_pool_capacity(), 4 * GB);
    assert_eq!(pools.total_capacity(), 20 * GB);
}

#[test]
fn test_working_memory_over_free() {
    let mut pools = WarmMemoryPools::rtx_5090();

    pools.allocate_working(GB).unwrap();

    // Free more than allocated - should not underflow
    pools.free_working(10 * GB).unwrap();

    // Should be at 0, not negative
    assert_eq!(pools.working_pool.allocated(), 0);
    assert_eq!(pools.available_working_bytes(), 8 * GB);
}

#[test]
fn test_zero_size_allocations() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Zero-size model allocation should succeed
    let result = pools.allocate_model("empty_model", 0, 0x1000);
    assert!(result.is_ok());

    // Zero-size working memory allocation should succeed
    let result = pools.allocate_working(0);
    assert!(result.is_ok());
}

#[test]
fn test_model_pool_exhaustion_exact() {
    let mut pools = WarmMemoryPools::rtx_5090();

    // Fill exactly to capacity
    pools.allocate_model("model1", 24 * GB, 0x1000).unwrap();

    // Even 1 byte more should fail
    let result = pools.allocate_model("model2", 1, 0x2000);
    assert!(result.is_err());
}
