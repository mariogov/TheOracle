//! Additional Unit Tests for Health Check Module
//!
//! More comprehensive tests for edge cases and utility methods.

use super::*;
use crate::warm::config::WarmConfig;
use crate::warm::handle::ModelHandle;
use crate::warm::memory_pool::WarmMemoryPools;
use crate::warm::registry::{SharedWarmRegistry, WarmModelRegistry};
use std::sync::{Arc, RwLock};

/// Helper to create a test config
fn test_config() -> WarmConfig {
    WarmConfig::default()
}

/// Helper to create a test handle
fn test_handle() -> ModelHandle {
    ModelHandle::new(0x1000_0000, 512 * 1024 * 1024, 0, 0xDEAD_BEEF)
}

// ============================================================================
// Additional Tests
// ============================================================================

#[test]
fn test_health_check_not_initialized_default() {
    let health = WarmHealthCheck::not_initialized();

    assert_eq!(health.status, WarmHealthStatus::NotInitialized);
    assert_eq!(health.models_total, 0);
    assert_eq!(health.warm_percentage(), 0.0);
    assert!(!health.is_healthy());
}

#[test]
fn test_health_check_default() {
    let health = WarmHealthCheck::default();
    assert_eq!(health.status, WarmHealthStatus::NotInitialized);
}

#[test]
fn test_health_status_display() {
    assert_eq!(format!("{}", WarmHealthStatus::Healthy), "healthy");
    assert_eq!(format!("{}", WarmHealthStatus::Loading), "loading");
    assert_eq!(format!("{}", WarmHealthStatus::Unhealthy), "unhealthy");
    assert_eq!(
        format!("{}", WarmHealthStatus::NotInitialized),
        "not_initialized"
    );
}

#[test]
fn test_vram_metrics() {
    let mut registry_inner = WarmModelRegistry::new();
    registry_inner
        .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
        .unwrap();

    let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
    let mut pools = WarmMemoryPools::new(test_config());

    // Allocate some VRAM
    pools
        .allocate_model("E1_Semantic", 500 * 1024 * 1024, 0x1000)
        .unwrap();

    let checker = WarmHealthChecker::new(registry, pools);
    let health = checker.check();

    // Verify VRAM metrics are populated
    assert_eq!(health.vram_allocated_bytes, 500 * 1024 * 1024);
    assert!(health.vram_available_bytes > 0);
    assert!(health.vram_total_bytes() > 0);
    assert!(health.vram_utilization() > 0.0);
}

#[test]
fn test_working_memory_metrics() {
    // Register at least one model so check() doesn't return early
    let mut registry_inner = WarmModelRegistry::new();
    registry_inner
        .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
        .unwrap();
    registry_inner.start_loading("E1_Semantic").unwrap();
    registry_inner.mark_validating("E1_Semantic").unwrap();
    registry_inner
        .mark_warm("E1_Semantic", test_handle())
        .unwrap();

    let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
    let mut pools = WarmMemoryPools::new(test_config());

    // Allocate some working memory
    pools.allocate_working(1024 * 1024 * 1024).unwrap(); // 1GB

    let checker = WarmHealthChecker::new(registry, pools);
    let health = checker.check();

    // Verify working memory metrics
    assert!(health.working_memory_allocated_bytes > 0);
    assert!(health.working_memory_available_bytes > 0);
}

#[test]
fn test_checker_clone() {
    let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));
    let pools = WarmMemoryPools::new(test_config());

    let checker1 = WarmHealthChecker::new(registry, pools);
    let checker2 = checker1.clone();

    // Both checkers should return the same status
    assert_eq!(checker1.status(), checker2.status());
}

#[test]
fn test_checker_accessors() {
    let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));
    let pools = WarmMemoryPools::new(test_config());

    let checker = WarmHealthChecker::new(registry.clone(), pools);

    // Verify accessors work
    assert!(std::sync::Arc::ptr_eq(checker.registry(), &registry));
    assert!(checker.memory_pools().is_within_budget());
}

#[test]
fn test_validating_state_counts_as_loading() {
    let mut registry_inner = WarmModelRegistry::new();

    registry_inner
        .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
        .unwrap();

    // Put model in Validating state
    registry_inner.start_loading("E1_Semantic").unwrap();
    registry_inner.mark_validating("E1_Semantic").unwrap();

    let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
    let pools = WarmMemoryPools::new(test_config());

    let checker = WarmHealthChecker::new(registry, pools);
    let health = checker.check();

    // Validating should count as loading
    assert_eq!(health.status, WarmHealthStatus::Loading);
    assert_eq!(health.models_loading, 1);
}

#[test]
fn test_warm_percentage_calculation() {
    let mut registry_inner = WarmModelRegistry::new();

    // 4 models, 2 warm
    for id in ["M1", "M2", "M3", "M4"] {
        registry_inner
            .register_model(id, 100 * 1024 * 1024, 768)
            .unwrap();
    }

    // Warm 2 models
    for id in ["M1", "M2"] {
        registry_inner.start_loading(id).unwrap();
        registry_inner.mark_validating(id).unwrap();
        registry_inner.mark_warm(id, test_handle()).unwrap();
    }

    let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
    let pools = WarmMemoryPools::new(test_config());

    let checker = WarmHealthChecker::new(registry, pools);
    let health = checker.check();

    // Should be 50% warm
    assert!((health.warm_percentage() - 50.0).abs() < 0.01);
}
