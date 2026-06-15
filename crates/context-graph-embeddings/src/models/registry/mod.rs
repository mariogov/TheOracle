//! ModelRegistry: Central manager for embedding model lifecycle.
//!
//! The registry is the single source of truth for loaded models. It provides:
//! - Lazy loading: models loaded on first access
//! - Thread-safe access: concurrent get_model() calls are safe
//! - Memory tracking: prevents loading models that exceed budget
//! - Per-model locks: serializes concurrent load requests for same model
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Errors propagate immediately with full context
//! - **FAIL FAST**: Invalid state = immediate EmbeddingError
//! - **NO MOCK DATA**: Tests use real ModelFactory implementations
//! - **THREAD SAFE**: All public methods safe for concurrent access
//!
//! # Example
//!
//! ```rust,no_run
//! use context_graph_embeddings::models::{ModelRegistry, ModelRegistryConfig};
//! use context_graph_embeddings::traits::ModelFactory;
//! use context_graph_embeddings::error::EmbeddingResult;
//! use context_graph_embeddings::types::ModelId;
//! use std::sync::Arc;
//!
//! async fn example(factory: Arc<dyn ModelFactory>) -> EmbeddingResult<()> {
//!     let config = ModelRegistryConfig::default();
//!     let registry = ModelRegistry::new(config, factory).await?;
//!
//!     // Lazy load on first access
//!     let model = registry.get_model(ModelId::Semantic).await?;
//!
//!     // Check stats
//!     let stats = registry.stats().await;
//!     println!("Loaded: {}, Memory: {}B", stats.loaded_count, stats.total_memory_bytes);
//!
//!     Ok(())
//! }
//! ```

mod config;
mod core;
mod loader;
mod operations;
mod stats;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_concurrent;
#[cfg(test)]
mod tests_mov;
#[cfg(test)]
mod tests_stats;
#[cfg(test)]
mod tests_unload;

// Re-export all public types for backwards compatibility
pub use config::ModelRegistryConfig;
pub use core::ModelRegistry;
pub use stats::{RegistryStats, RegistryStatsInternal};
