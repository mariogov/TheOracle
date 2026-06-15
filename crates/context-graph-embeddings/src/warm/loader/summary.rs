//! Loading summary types for the warm model loader.
//!
//! Provides diagnostic information about the loading state of all models,
//! memory usage, and timing information.

use std::collections::HashMap;
use std::time::Duration;

use crate::warm::state::WarmModelState;

use super::helpers::format_bytes;

/// Summary of the warm loading operation.
///
/// Provides diagnostic information about the loading state of all models,
/// memory usage, and timing information.
#[derive(Debug, Clone)]
pub struct LoadingSummary {
    /// Total number of registered models.
    pub total_models: usize,
    /// Number of models in Warm state.
    pub models_warm: usize,
    /// Number of models in Failed state.
    pub models_failed: usize,
    /// Number of models currently Loading.
    pub models_loading: usize,
    /// Total VRAM allocated for models (bytes).
    pub total_vram_allocated: usize,
    /// Total time spent loading (if completed).
    pub loading_duration: Option<Duration>,
    /// Current state of each model.
    pub model_states: HashMap<String, WarmModelState>,
}

impl LoadingSummary {
    /// Create an empty summary.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            total_models: 0,
            models_warm: 0,
            models_failed: 0,
            models_loading: 0,
            total_vram_allocated: 0,
            loading_duration: None,
            model_states: HashMap::new(),
        }
    }

    /// Check if all models are warm.
    #[must_use]
    pub fn all_warm(&self) -> bool {
        self.total_models > 0 && self.models_warm == self.total_models
    }

    /// Check if any model failed.
    #[must_use]
    pub fn any_failed(&self) -> bool {
        self.models_failed > 0
    }

    /// Get the percentage of models that are warm.
    #[must_use]
    pub fn warm_percentage(&self) -> f64 {
        if self.total_models == 0 {
            return 0.0;
        }
        (self.models_warm as f64 / self.total_models as f64) * 100.0
    }

    /// Format VRAM as human-readable string.
    #[must_use]
    pub fn vram_allocated_string(&self) -> String {
        format_bytes(self.total_vram_allocated)
    }
}

impl Default for LoadingSummary {
    fn default() -> Self {
        Self::empty()
    }
}
