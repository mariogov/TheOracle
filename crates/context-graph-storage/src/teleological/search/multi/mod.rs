//! Multi-embedder parallel search for HNSW indexes.
//!
//! # Overview
//!
//! Searches MULTIPLE embedder indexes in parallel using rayon, combining
//! results from different semantic spaces for comprehensive retrieval.
//! This is Stage 3/5 of the 5-stage teleological retrieval pipeline.
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**
//!
//! All errors are fatal. No recovery attempts. This ensures:
//! - Bugs are caught early in development
//! - Data integrity is preserved
//! - Clear error messages for debugging
//!
//! # Module Structure
//!
//! - `types` - Configuration, result types, and aggregation structures
//! - `builder` - MultiSearchBuilder for fluent API
//! - `executor` - MultiEmbedderSearch execution engine

mod builder;
mod executor;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use builder::MultiSearchBuilder;
pub use executor::MultiEmbedderSearch;
pub use types::{
    AggregatedHit, AggregationStrategy, MultiEmbedderSearchConfig, MultiEmbedderSearchResults,
    NormalizationStrategy, PerEmbedderResults,
};
