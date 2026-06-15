//! Single model embedding output with validation and normalization.
//!
//! This module provides the `ModelEmbedding` struct which represents
//! the output from a single embedding model in the 13-model pipeline.
//!
//! # Submodules
//! - `types` - Core struct definition and constructors
//! - `validation` - Validation logic for embeddings
//! - `operations` - Mathematical operations (normalization, similarity)
//! - `tensor` - GPU tensor conversions (requires `candle` feature)

mod operations;
mod tensor;
mod types;
mod validation;

#[cfg(test)]
mod tests;

// Re-export the main type for backwards compatibility
pub use types::ModelEmbedding;
