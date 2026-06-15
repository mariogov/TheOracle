//! Configuration types for ModelRegistry.
//!
//! This module provides the `ModelRegistryConfig` struct for configuring registry behavior
//! including memory limits, concurrent load limits, and preloading options.

use std::collections::HashSet;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

/// Configuration for ModelRegistry.
///
/// Controls registry behavior including memory limits, preloading, and logging.
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::models::ModelRegistryConfig;
/// use context_graph_embeddings::types::ModelId;
///
/// let config = ModelRegistryConfig {
///     max_concurrent_loads: 2,
///     memory_budget_bytes: 32_000_000_000, // 32GB
///     preload_models: vec![ModelId::Semantic, ModelId::Code],
///     enable_debug_logging: true,
/// };
///
/// assert_eq!(config.max_concurrent_loads, 2);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct ModelRegistryConfig {
    /// Maximum number of models that can load concurrently.
    /// Default: 4
    pub max_concurrent_loads: usize,

    /// Total memory budget in bytes.
    /// Default: 32GB (RTX 5090 VRAM)
    pub memory_budget_bytes: usize,

    /// Models to preload on initialize().
    /// Default: empty (all lazy loaded)
    pub preload_models: Vec<ModelId>,

    /// Enable detailed debug logging.
    /// Default: false
    pub enable_debug_logging: bool,
}

impl Default for ModelRegistryConfig {
    fn default() -> Self {
        Self {
            max_concurrent_loads: 4,
            memory_budget_bytes: 32_000_000_000, // 32GB RTX 5090
            preload_models: Vec::new(),
            enable_debug_logging: false,
        }
    }
}

impl ModelRegistryConfig {
    /// Create config for RTX 5090 (32GB VRAM).
    pub fn rtx_5090() -> Self {
        Self {
            memory_budget_bytes: 32_000_000_000,
            ..Default::default()
        }
    }

    /// Create config for RTX 4090 (24GB VRAM).
    pub fn rtx_4090() -> Self {
        Self {
            memory_budget_bytes: 24_000_000_000,
            ..Default::default()
        }
    }

    /// Create config for testing with limited memory.
    pub fn testing(budget_bytes: usize) -> Self {
        Self {
            memory_budget_bytes: budget_bytes,
            enable_debug_logging: true,
            ..Default::default()
        }
    }

    /// Validate configuration.
    ///
    /// # Errors
    /// - `ConfigError` if max_concurrent_loads is 0
    /// - `ConfigError` if memory_budget_bytes is 0
    /// - `ConfigError` if preload_models contains duplicates
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.max_concurrent_loads == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "max_concurrent_loads must be greater than 0".to_string(),
            });
        }

        if self.memory_budget_bytes == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "memory_budget_bytes must be greater than 0".to_string(),
            });
        }

        // Check for duplicate preload models
        let mut seen = HashSet::new();
        for model_id in &self.preload_models {
            if !seen.insert(*model_id) {
                return Err(EmbeddingError::ConfigError {
                    message: format!("duplicate model in preload_models: {:?}", model_id),
                });
            }
        }

        Ok(())
    }
}
