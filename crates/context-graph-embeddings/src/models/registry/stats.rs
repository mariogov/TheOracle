//! Statistics types for ModelRegistry.
//!
//! This module provides types for tracking registry statistics including
//! load counts, cache hits, and memory usage.

/// Public statistics snapshot.
///
/// Immutable snapshot of registry statistics at a point in time.
#[derive(Debug, Clone, Default)]
pub struct RegistryStats {
    /// Number of currently loaded models.
    pub loaded_count: usize,
    /// Total memory usage in bytes.
    pub total_memory_bytes: usize,
    /// Total number of model loads.
    pub load_count: u64,
    /// Total number of model unloads.
    pub unload_count: u64,
    /// Cache hits (get_model for already loaded model).
    pub cache_hits: u64,
    /// Failed load attempts.
    pub load_failures: u64,
}

/// Internal mutable statistics.
#[derive(Debug, Default)]
pub struct RegistryStatsInternal {
    /// Total number of model loads.
    pub load_count: u64,
    /// Total number of model unloads.
    pub unload_count: u64,
    /// Cache hits (get_model for already loaded model).
    pub cache_hits: u64,
    /// Failed load attempts.
    pub load_failures: u64,
}
