//! Health Check Result
//!
//! Contains the detailed health check result struct with comprehensive
//! information about the warm loading system state.

use std::time::{Duration, Instant};

use super::status::WarmHealthStatus;

/// Detailed health check result.
///
/// Contains comprehensive information about the warm loading system state,
/// including model counts, VRAM usage, and error details.
#[derive(Debug, Clone)]
pub struct WarmHealthCheck {
    /// Overall health status.
    pub status: WarmHealthStatus,

    /// Total number of registered models.
    pub models_total: usize,

    /// Number of models in `Warm` state.
    pub models_warm: usize,

    /// Number of models in `Loading` or `Validating` state.
    pub models_loading: usize,

    /// Number of models in `Failed` state.
    pub models_failed: usize,

    /// Number of models in `Pending` state.
    pub models_pending: usize,

    /// Total VRAM allocated for models (bytes).
    pub vram_allocated_bytes: usize,

    /// Available VRAM for model allocations (bytes).
    pub vram_available_bytes: usize,

    /// Working memory allocated (bytes).
    pub working_memory_allocated_bytes: usize,

    /// Working memory available (bytes).
    pub working_memory_available_bytes: usize,

    /// System uptime since checker creation.
    pub uptime: Option<Duration>,

    /// Timestamp of this health check.
    pub last_check: Instant,

    /// Error messages from failed models.
    ///
    /// Format: `["model_id: error message", ...]`
    pub error_messages: Vec<String>,
}

impl WarmHealthCheck {
    /// Create an empty health check result for uninitialized state.
    #[must_use]
    pub fn not_initialized() -> Self {
        Self {
            status: WarmHealthStatus::NotInitialized,
            models_total: 0,
            models_warm: 0,
            models_loading: 0,
            models_failed: 0,
            models_pending: 0,
            vram_allocated_bytes: 0,
            vram_available_bytes: 0,
            working_memory_allocated_bytes: 0,
            working_memory_available_bytes: 0,
            uptime: None,
            last_check: Instant::now(),
            error_messages: Vec::new(),
        }
    }

    /// Check if the system is healthy.
    #[inline]
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.status.is_healthy()
    }

    /// Get warm model percentage (0.0 - 100.0).
    #[must_use]
    pub fn warm_percentage(&self) -> f64 {
        if self.models_total == 0 {
            return 0.0;
        }
        (self.models_warm as f64 / self.models_total as f64) * 100.0
    }

    /// Get total VRAM capacity (allocated + available).
    #[must_use]
    pub fn vram_total_bytes(&self) -> usize {
        self.vram_allocated_bytes
            .saturating_add(self.vram_available_bytes)
    }

    /// Get VRAM utilization percentage (0.0 - 1.0).
    #[must_use]
    pub fn vram_utilization(&self) -> f64 {
        let total = self.vram_total_bytes();
        if total == 0 {
            return 0.0;
        }
        self.vram_allocated_bytes as f64 / total as f64
    }
}

impl Default for WarmHealthCheck {
    fn default() -> Self {
        Self::not_initialized()
    }
}
