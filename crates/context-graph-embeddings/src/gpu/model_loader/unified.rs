//! Unified model loader that bridges ModelSlotManager with Candle model loading.
//!
//! # Architecture (TASK-CORE-012)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        UnifiedModelLoader                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  ModelSlotManager    │   GpuModelLoader   │     ModelFactory           │
//! │  (VRAM budget)       │   (Candle loader)  │     (Model creation)       │
//! │                      │                    │                            │
//! │  ┌─────────────┐     │  ┌──────────────┐  │  ┌───────────────────┐     │
//! │  │ 8GB Budget  │────▶│  │ safetensors  │──▶│  │ BertWeights etc  │     │
//! │  │ LRU Evict   │     │  │ VarBuilder   │  │  │ Box<EmbeddingModel>│    │
//! │  └─────────────┘     │  └──────────────┘  │  └───────────────────┘     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Memory Budget**: 8GB total for all 13 models (quantized)
//! - **LRU Eviction**: Automatic eviction when memory pressure is critical
//! - **Candle Integration**: Load safetensors via GpuModelLoader
//! - **Type Conversion**: Seamless Embedder ↔ ModelId conversion

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use context_graph_core::teleological::embedder::Embedder;

use crate::gpu::memory::{MemoryError, ModelSlotManager};
use crate::types::ModelId;

use super::error::ModelLoadError;
use super::loader::GpuModelLoader;
use super::weights::BertWeights;

/// Configuration for the UnifiedModelLoader.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Base directory containing model subdirectories.
    /// e.g., "/home/user/models" containing "semantic/", "code/", etc.
    pub models_dir: PathBuf,

    /// Maximum memory budget in bytes (default: 8GB).
    pub memory_budget: usize,

    /// Enable automatic LRU eviction when memory pressure is critical.
    pub enable_auto_eviction: bool,

    /// Preload these models on initialization (empty = load on demand).
    pub preload_models: Vec<ModelId>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("./models"),
            memory_budget: 8 * 1024 * 1024 * 1024, // 8GB
            enable_auto_eviction: true,
            preload_models: Vec::new(),
        }
    }
}

impl LoaderConfig {
    /// Create config with custom models directory.
    pub fn with_models_dir(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            ..Default::default()
        }
    }

    /// Set memory budget in bytes.
    pub fn with_budget(mut self, budget_bytes: usize) -> Self {
        self.memory_budget = budget_bytes;
        self
    }

    /// Enable/disable automatic LRU eviction.
    pub fn with_auto_eviction(mut self, enabled: bool) -> Self {
        self.enable_auto_eviction = enabled;
        self
    }

    /// Set models to preload on initialization.
    pub fn with_preload(mut self, models: Vec<ModelId>) -> Self {
        self.preload_models = models;
        self
    }

    /// Get the path to a specific model's directory.
    pub fn model_path(&self, model_id: ModelId) -> PathBuf {
        self.models_dir.join(model_id.as_str())
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), LoaderConfigError> {
        if self.memory_budget == 0 {
            return Err(LoaderConfigError::ZeroBudget);
        }

        if !self.models_dir.exists() {
            return Err(LoaderConfigError::ModelsDirectoryNotFound {
                path: self.models_dir.clone(),
            });
        }

        Ok(())
    }
}

/// Errors from configuration validation.
#[derive(Debug, thiserror::Error)]
pub enum LoaderConfigError {
    #[error("Memory budget cannot be zero")]
    ZeroBudget,

    #[error("Models directory not found: {path}")]
    ModelsDirectoryNotFound { path: PathBuf },
}

/// Unified model loader that manages model loading with memory constraints.
///
/// Bridges the gap between:
/// - `ModelSlotManager`: Memory budget and LRU eviction
/// - `GpuModelLoader`: Actual Candle model loading from safetensors
///
/// # Thread Safety
///
/// This struct is thread-safe via internal `RwLock` on the slot manager.
/// Multiple threads can call `load_model()` concurrently.
pub struct UnifiedModelLoader {
    /// Configuration for the loader.
    config: LoaderConfig,

    /// Thread-safe slot manager for memory tracking.
    slot_manager: Arc<RwLock<ModelSlotManager>>,

    /// Candle-based GPU model loader.
    gpu_loader: GpuModelLoader,

    /// Loaded BERT weights, keyed by Embedder.
    loaded_weights: Arc<RwLock<std::collections::HashMap<Embedder, BertWeights>>>,
}

impl UnifiedModelLoader {
    /// Create a new UnifiedModelLoader with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Configuration validation fails
    /// - GPU initialization fails
    pub fn new(config: LoaderConfig) -> Result<Self, UnifiedLoaderError> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| UnifiedLoaderError::ConfigError { source: e })?;

        // Initialize GPU loader
        let gpu_loader = GpuModelLoader::new().map_err(|e| UnifiedLoaderError::GpuInitFailed {
            message: e.to_string(),
        })?;

        // Create slot manager with configured budget
        let slot_manager = ModelSlotManager::with_budget(config.memory_budget);

        Ok(Self {
            config,
            slot_manager: Arc::new(RwLock::new(slot_manager)),
            gpu_loader,
            loaded_weights: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Load a model by ModelId, managing memory automatically.
    ///
    /// # Process
    ///
    /// 1. Convert ModelId to Embedder for slot tracking
    /// 2. Check if model is already loaded
    /// 3. Estimate memory requirement
    /// 4. If over budget and auto-eviction enabled, evict LRU models
    /// 5. Allocate slot in ModelSlotManager
    /// 6. Load weights via GpuModelLoader
    /// 7. Store in loaded_weights map
    ///
    /// # Returns
    ///
    /// Reference to the loaded `BertWeights`.
    ///
    /// # Errors
    ///
    /// - `UnifiedLoaderError::OutOfMemory` if budget exceeded and can't evict
    /// - `UnifiedLoaderError::ModelLoadFailed` if Candle loading fails
    pub fn load_model(&self, model_id: ModelId) -> Result<(), UnifiedLoaderError> {
        let embedder: Embedder = model_id.into();

        // Check if already loaded
        {
            let loaded = self
                .loaded_weights
                .read()
                .map_err(|_| UnifiedLoaderError::LockPoisoned)?;
            if loaded.contains_key(&embedder) {
                tracing::debug!(model_id = ?model_id, "Model already loaded, skipping");
                return Ok(());
            }
        }

        // Get model path
        let model_path = self.config.model_path(model_id);
        if !model_path.exists() {
            return Err(UnifiedLoaderError::ModelNotFound {
                model_id,
                path: model_path,
            });
        }

        // Estimate memory requirement (conservative estimate based on config.json)
        let memory_estimate = self.estimate_model_memory(&model_path)?;

        tracing::info!(
            model_id = ?model_id,
            embedder = ?embedder,
            memory_mb = memory_estimate / (1024 * 1024),
            "Loading model with memory estimate"
        );

        // Allocate slot (with eviction if enabled and needed)
        self.allocate_slot_for_model(embedder, memory_estimate)?;

        // Actually load the model via Candle
        let weights = self
            .gpu_loader
            .load_bert_weights(&model_path)
            .map_err(|e| {
                // Deallocate on failure
                let _ = self.deallocate_slot(embedder);
                UnifiedLoaderError::ModelLoadFailed {
                    model_id,
                    source: e,
                }
            })?;

        // Store the loaded weights
        {
            let mut loaded = self
                .loaded_weights
                .write()
                .map_err(|_| UnifiedLoaderError::LockPoisoned)?;
            loaded.insert(embedder, weights);
        }

        // Update slot with actual VRAM usage
        self.update_slot_with_actual_size(embedder)?;

        tracing::info!(
            model_id = ?model_id,
            embedder = ?embedder,
            "Model loaded successfully"
        );

        Ok(())
    }

    /// Unload a model, freeing its memory slot.
    pub fn unload_model(&self, model_id: ModelId) -> Result<(), UnifiedLoaderError> {
        let embedder: Embedder = model_id.into();

        // Remove from loaded weights
        {
            let mut loaded = self
                .loaded_weights
                .write()
                .map_err(|_| UnifiedLoaderError::LockPoisoned)?;
            loaded.remove(&embedder);
        }

        // Deallocate slot
        self.deallocate_slot(embedder)?;

        tracing::info!(model_id = ?model_id, "Model unloaded");
        Ok(())
    }

    /// Check if a model is loaded.
    pub fn is_loaded(&self, model_id: ModelId) -> bool {
        let embedder: Embedder = model_id.into();
        self.loaded_weights
            .read()
            .map(|loaded| loaded.contains_key(&embedder))
            .unwrap_or(false)
    }

    /// List all currently loaded models.
    pub fn loaded_models(&self) -> Result<Vec<ModelId>, UnifiedLoaderError> {
        let loaded = self
            .loaded_weights
            .read()
            .map_err(|_| UnifiedLoaderError::LockPoisoned)?;

        Ok(loaded.keys().copied().map(ModelId::from).collect())
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Estimate memory for a model based on config.json.
    fn estimate_model_memory(&self, model_path: &Path) -> Result<usize, UnifiedLoaderError> {
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            // Use conservative default estimate
            return Ok(500 * 1024 * 1024); // 500MB default
        }

        // Parse config to estimate size
        let config = self.gpu_loader.load_config(model_path).map_err(|e| {
            UnifiedLoaderError::ModelLoadFailed {
                model_id: ModelId::Semantic, // placeholder
                source: e,
            }
        })?;

        // Rough estimate: parameters * 4 bytes (FP32)
        // Embedding: vocab_size * hidden_size
        // Attention: 4 * hidden_size * hidden_size per layer
        // FFN: 2 * hidden_size * intermediate_size per layer
        let embedding_params = config.vocab_size * config.hidden_size;
        let attention_per_layer = 4 * config.hidden_size * config.hidden_size;
        let ffn_per_layer = 2 * config.hidden_size * config.intermediate_size;
        let layer_params = (attention_per_layer + ffn_per_layer) * config.num_hidden_layers;
        let total_params = embedding_params + layer_params;

        // FP32 = 4 bytes, add 20% overhead for activations/buffers
        let estimated_bytes = (total_params * 4) as f64 * 1.2;

        Ok(estimated_bytes as usize)
    }

    /// Allocate a slot for the model, evicting LRU if needed.
    fn allocate_slot_for_model(
        &self,
        embedder: Embedder,
        size_bytes: usize,
    ) -> Result<(), UnifiedLoaderError> {
        let mut slot_manager = self
            .slot_manager
            .write()
            .map_err(|_| UnifiedLoaderError::LockPoisoned)?;

        if self.config.enable_auto_eviction {
            // Use allocate_with_eviction for automatic LRU eviction
            let evicted = slot_manager
                .allocate_with_eviction(embedder, size_bytes)
                .map_err(|e| match e {
                    MemoryError::OutOfMemory {
                        requested,
                        available,
                    } => UnifiedLoaderError::OutOfMemory {
                        requested,
                        available,
                        budget: slot_manager.budget(),
                    },
                    MemoryError::LockPoisoned => UnifiedLoaderError::LockPoisoned,
                })?;

            // Remove evicted models from loaded_weights
            if !evicted.is_empty() {
                drop(slot_manager); // Release lock before acquiring loaded_weights lock
                let mut loaded = self
                    .loaded_weights
                    .write()
                    .map_err(|_| UnifiedLoaderError::LockPoisoned)?;
                for e in evicted {
                    tracing::info!(evicted = ?e, "Evicted model due to memory pressure");
                    loaded.remove(&e);
                }
            }
        } else {
            // Direct allocation without eviction
            slot_manager
                .allocate_slot(embedder, size_bytes)
                .map_err(|e| match e {
                    MemoryError::OutOfMemory {
                        requested,
                        available,
                    } => UnifiedLoaderError::OutOfMemory {
                        requested,
                        available,
                        budget: slot_manager.budget(),
                    },
                    MemoryError::LockPoisoned => UnifiedLoaderError::LockPoisoned,
                })?;
        }

        Ok(())
    }

    /// Deallocate a slot.
    fn deallocate_slot(&self, embedder: Embedder) -> Result<usize, UnifiedLoaderError> {
        let mut slot_manager = self
            .slot_manager
            .write()
            .map_err(|_| UnifiedLoaderError::LockPoisoned)?;
        Ok(slot_manager.deallocate_slot(&embedder))
    }

    /// Update slot with actual VRAM size from loaded weights.
    fn update_slot_with_actual_size(&self, embedder: Embedder) -> Result<(), UnifiedLoaderError> {
        let loaded = self
            .loaded_weights
            .read()
            .map_err(|_| UnifiedLoaderError::LockPoisoned)?;

        if let Some(weights) = loaded.get(&embedder) {
            let actual_size = weights.vram_bytes();
            drop(loaded);

            let mut slot_manager = self
                .slot_manager
                .write()
                .map_err(|_| UnifiedLoaderError::LockPoisoned)?;

            // Re-allocate with actual size (allocate_slot updates existing slots)
            slot_manager
                .allocate_slot(embedder, actual_size)
                .map_err(|e| match e {
                    MemoryError::OutOfMemory {
                        requested,
                        available,
                    } => UnifiedLoaderError::OutOfMemory {
                        requested,
                        available,
                        budget: slot_manager.budget(),
                    },
                    MemoryError::LockPoisoned => UnifiedLoaderError::LockPoisoned,
                })?;
        }

        Ok(())
    }
}

/// Errors from the unified model loader.
#[derive(Debug, thiserror::Error)]
pub enum UnifiedLoaderError {
    /// Configuration validation failed.
    #[error("Configuration error: {source}")]
    ConfigError {
        #[source]
        source: LoaderConfigError,
    },

    /// GPU initialization failed.
    #[error("GPU initialization failed: {message}")]
    GpuInitFailed { message: String },

    /// Model directory not found.
    #[error("Model not found: {model_id:?} at {path}")]
    ModelNotFound { model_id: ModelId, path: PathBuf },

    /// Out of memory (cannot allocate).
    #[error("Out of memory: requested {requested} bytes, available {available} bytes (budget: {budget} bytes)")]
    OutOfMemory {
        requested: usize,
        available: usize,
        budget: usize,
    },

    /// Model loading failed.
    #[error("Model load failed for {model_id:?}: {source}")]
    ModelLoadFailed {
        model_id: ModelId,
        #[source]
        source: ModelLoadError,
    },

    /// Internal lock poisoned.
    #[error("Internal lock poisoned")]
    LockPoisoned,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_config_default() {
        let config = LoaderConfig::default();
        assert_eq!(config.memory_budget, 8 * 1024 * 1024 * 1024);
        assert!(config.enable_auto_eviction);
        assert!(config.preload_models.is_empty());
        println!("[PASS] LoaderConfig::default() has correct values");
    }

    #[test]
    fn test_loader_config_builder() {
        let config = LoaderConfig::with_models_dir("/tmp/models")
            .with_budget(4 * 1024 * 1024 * 1024)
            .with_auto_eviction(false)
            .with_preload(vec![ModelId::Semantic, ModelId::Code]);

        assert_eq!(config.models_dir, PathBuf::from("/tmp/models"));
        assert_eq!(config.memory_budget, 4 * 1024 * 1024 * 1024);
        assert!(!config.enable_auto_eviction);
        assert_eq!(config.preload_models.len(), 2);
        println!("[PASS] LoaderConfig builder methods work correctly");
    }

    #[test]
    fn test_loader_config_model_path() {
        let config = LoaderConfig::with_models_dir("/home/user/models");
        let path = config.model_path(ModelId::Semantic);
        assert_eq!(path, PathBuf::from("/home/user/models/semantic"));

        let code_path = config.model_path(ModelId::Code);
        assert_eq!(code_path, PathBuf::from("/home/user/models/code"));
        println!("[PASS] LoaderConfig::model_path() generates correct paths");
    }

    #[test]
    fn test_loader_config_validation_zero_budget() {
        let config = LoaderConfig::default().with_budget(0);
        let result = config.validate();
        assert!(matches!(result, Err(LoaderConfigError::ZeroBudget)));
        println!("[PASS] LoaderConfig validation rejects zero budget");
    }

    #[test]
    fn test_loader_config_validation_missing_dir() {
        let config = LoaderConfig::with_models_dir("/nonexistent/path/that/does/not/exist");
        let result = config.validate();
        assert!(matches!(
            result,
            Err(LoaderConfigError::ModelsDirectoryNotFound { .. })
        ));
        println!("[PASS] LoaderConfig validation rejects missing directory");
    }

    // Note: Tests requiring actual GPU/model loading are in integration tests
    // since they require the models directory and CUDA hardware.
}
