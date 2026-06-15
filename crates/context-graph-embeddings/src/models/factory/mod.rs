//! Default model factory implementation.
//!
//! This module provides `DefaultModelFactory`, the production factory for creating
//! all 12 embedding model instances with proper configuration.
//!
//! # Architecture
//!
//! ```text
//! DefaultModelFactory
//! ├── models_dir: PathBuf (base path for pretrained model files)
//! ├── gpu_config: GpuConfig (GPU settings: device, memory, features)
//! └── create_model() -> Box<dyn EmbeddingModel>
//! ```
//!
//! # Model Categories
//!
//! ## Pretrained Models (require model files)
//! - E1: SemanticModel (intfloat/e5-large-v2)
//! - E5: CausalModel (nomic-embed-text-v1.5)
//! - E6: SparseModel (SPLADE)
//! - E7: CodeModel (CodeBERT)
//! - E8: GraphModel (sentence-transformers)
//! - E10: ContextualModel (intfloat/e5-base-v2)
//! - E11: KeplerModel (KEPLER RoBERTa + TransE)
//! - E12: LateInteractionModel (ColBERT)
//!
//! ## Custom Models (lightweight, no model files)
//! - E2: TemporalRecentModel (decay-based)
//! - E3: TemporalPeriodicModel (fourier features)
//! - E4: TemporalPositionalModel (sinusoidal encoding)
//! - E9: HdcModel (hyperdimensional computing)
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Invalid config returns `EmbeddingError::ConfigError`
//! - **FAIL FAST**: Unknown ModelId returns `EmbeddingError::ModelNotFound`
//! - **THREAD SAFE**: Factory is `Send + Sync` for concurrent access
//!
//! # Example
//!
//! ```rust
//! use context_graph_embeddings::models::DefaultModelFactory;
//! use context_graph_embeddings::config::GpuConfig;
//! use context_graph_embeddings::traits::{ModelFactory, SingleModelConfig};
//! use context_graph_embeddings::types::ModelId;
//! use std::path::PathBuf;
//!
//! // Create factory
//! let factory = DefaultModelFactory::new(
//!     PathBuf::from("./models"),
//!     GpuConfig::default(),
//! );
//!
//! // Configure model placement
//! let config = SingleModelConfig::cuda_fp16();
//!
//! // Check memory requirements before creating
//! let memory_needed = factory.estimate_memory(ModelId::Semantic);
//! assert!(memory_needed > 0);
//!
//! // Create model (requires model files at ./models/semantic/)
//! // let model = factory.create_model(ModelId::Semantic, &config)?;
//! ```

mod default_factory;
mod trait_impl;

#[cfg(test)]
mod tests;

// Re-export for backwards compatibility
pub use default_factory::DefaultModelFactory;
