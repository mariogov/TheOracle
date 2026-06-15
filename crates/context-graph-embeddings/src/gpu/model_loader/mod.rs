//! GPU ModelLoader for loading pretrained BERT models from safetensors.
//!
//! # Architecture
//!
//! This module provides GPU-accelerated model loading via Candle's VarBuilder.
//! It loads safetensors files from local model directories and constructs
//! complete BERT architecture components for embedding generation.
//!
//! # Module Structure (TASK-CORE-012)
//!
//! ## Core Components
//!
//! - [`GpuModelLoader`] - Low-level Candle loader for safetensors
//! - [`UnifiedModelLoader`] - High-level loader with memory management
//! - [`LoaderConfig`] - Configuration for unified loading
//!
//! ## Integration Flow
//!
//! ```text
//! LoaderConfig
//!      │
//!      ▼
//! UnifiedModelLoader ──────┬────────────────┐
//!      │                   │                │
//!      ▼                   ▼                ▼
//! ModelSlotManager    GpuModelLoader   BertWeights
//! (8GB budget, LRU)   (Candle/CUDA)    (Model data)
//! ```
//!
//! # Supported Architectures
//!
//! | Model Type | Architecture | Example |
//! |------------|--------------|---------|
//! | BERT | BertModel | e5-large-v2, all-MiniLM-L6-v2 |
//! | MPNet | MPNetModel | all-mpnet-base-v2 |
//!
//! # Usage
//!
//! ```rust,no_run
//! use context_graph_embeddings::gpu::{UnifiedModelLoader, LoaderConfig};
//! use context_graph_embeddings::types::ModelId;
//!
//! // Configure loader with models directory
//! let config = LoaderConfig::with_models_dir("/home/user/models")
//!     .with_budget(8 * 1024 * 1024 * 1024)  // 8GB
//!     .with_auto_eviction(true);
//!
//! // Create unified loader (initializes GPU automatically)
//! let loader = UnifiedModelLoader::new(config).expect("Loader init");
//!
//! // Load a model (manages memory automatically)
//! loader.load_model(ModelId::Semantic).expect("Load failed");
//!
//! // Check if model is loaded
//! assert!(loader.is_loaded(ModelId::Semantic));
//! ```

mod batch_loader;
mod config;
mod embedding_loader;
mod error;
mod layer_loader;
mod loader;
mod tensor_utils;
mod unified;
mod weights;

// Re-export all public types
pub use config::BertConfig;
pub use error::ModelLoadError;
pub use loader::GpuModelLoader;
pub use unified::{LoaderConfig, LoaderConfigError, UnifiedLoaderError, UnifiedModelLoader};
pub use weights::{
    AttentionWeights, BertWeights, EmbeddingWeights, EncoderLayerWeights, FfnWeights, PoolerWeights,
};
