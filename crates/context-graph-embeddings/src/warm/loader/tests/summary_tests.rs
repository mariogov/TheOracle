//! Tests for loading summary, helpers, and constants.

use std::collections::HashMap;

use crate::warm::config::WarmConfig;
use crate::warm::registry::EMBEDDING_MODEL_IDS;

use super::super::constants::{GB, MODEL_SIZES};
use super::super::engine::WarmLoader;
use super::super::helpers::format_bytes;
use super::super::summary::LoadingSummary;

/// Create a test config that doesn't require real files.
#[allow(clippy::field_reassign_with_default)]
fn test_config() -> WarmConfig {
    let mut config = WarmConfig::default();
    config.enable_test_inference = true;
    config
}

// ============================================================================
// Format Bytes Tests
// ============================================================================

#[test]
fn test_format_bytes() {
    assert_eq!(format_bytes(0), "0B");
    assert_eq!(format_bytes(512), "512B");
    assert_eq!(format_bytes(1024), "1.00KB");
    assert_eq!(format_bytes(1024 * 1024), "1.00MB");
    assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00GB");
    assert_eq!(format_bytes(32 * 1024 * 1024 * 1024), "32.00GB");
}

// ============================================================================
// Loading Summary Tests
// ============================================================================

#[test]
fn test_loading_summary_empty() {
    let summary = LoadingSummary::empty();
    assert_eq!(summary.total_models, 0);
    assert_eq!(summary.models_warm, 0);
    assert!(!summary.all_warm());
    assert!(!summary.any_failed());
    assert_eq!(summary.warm_percentage(), 0.0);
}

#[test]
fn test_loading_summary_vram_string() {
    let mut summary = LoadingSummary::empty();
    summary.total_vram_allocated = 24 * GB;
    assert_eq!(summary.vram_allocated_string(), "24.00GB");
}

// ============================================================================
// Memory Pools Tests
// ============================================================================

#[test]
fn test_memory_pools_initialized() {
    let config = test_config();
    let loader = WarmLoader::new(config).expect("Failed to create loader");

    let pools = loader.memory_pools();
    assert_eq!(pools.model_pool_capacity(), 24 * GB);
    assert_eq!(pools.working_pool_capacity(), 8 * GB);
    assert!(pools.is_within_budget());
}

// ============================================================================
// Constants Tests
// ============================================================================

#[test]
fn test_model_sizes_constants() {
    // Verify we have sizes for all models
    let size_map: HashMap<&str, usize> = MODEL_SIZES.iter().copied().collect();

    for model_id in EMBEDDING_MODEL_IDS {
        assert!(
            size_map.contains_key(model_id),
            "Missing size for {}",
            model_id
        );
    }

    // Verify total is reasonable (should fit in 24GB)
    let total_size: usize = MODEL_SIZES.iter().map(|(_, s)| s).sum();
    assert!(total_size < 24 * GB, "Total model size exceeds budget");
}
