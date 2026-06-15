//! Type definitions for memory pool management.
//!
//! This module contains the core data structures used for tracking
//! VRAM allocations in the warm loading system.

use std::time::Instant;

/// Tracks a single model's VRAM allocation.
///
/// Each allocation records the model identifier, VRAM pointer, size,
/// and timestamp for diagnostics and lifecycle management.
#[derive(Debug, Clone)]
pub struct ModelAllocation {
    /// Unique model identifier (e.g., "E1_Semantic").
    pub model_id: String,
    /// Raw VRAM pointer from CUDA allocation.
    ///
    /// This is an opaque handle; the actual memory management is
    /// performed by the CUDA runtime.
    pub vram_ptr: u64,
    /// Size of allocation in bytes.
    pub size_bytes: usize,
    /// Timestamp when allocation was made.
    ///
    /// Used for diagnostics and debugging memory fragmentation.
    pub allocated_at: Instant,
}

impl ModelAllocation {
    /// Create a new model allocation record.
    #[must_use]
    pub fn new(model_id: String, vram_ptr: u64, size_bytes: usize) -> Self {
        Self {
            model_id,
            vram_ptr,
            size_bytes,
            allocated_at: Instant::now(),
        }
    }

    /// Get the age of this allocation in seconds.
    #[must_use]
    pub fn age_secs(&self) -> f64 {
        self.allocated_at.elapsed().as_secs_f64()
    }
}
