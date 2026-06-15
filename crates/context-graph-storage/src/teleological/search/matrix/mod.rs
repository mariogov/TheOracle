//! Matrix strategy search with cross-embedder correlations.
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
//! # Overview
//!
//! Matrix strategy search applies a full 13x13 weight matrix for search. This enables
//! cross-embedder correlation analysis where off-diagonal weights capture relationships
//! between different embedding spaces.
//!
//! # Example
//!
//! ```no_run
//! use context_graph_storage::teleological::search::{
//!     MatrixStrategySearch, SearchMatrix, MatrixSearchBuilder,
//! };
//! use context_graph_storage::teleological::indexes::EmbedderIndexRegistry;
//! use std::sync::Arc;
//! use std::collections::HashMap;
//! use context_graph_storage::teleological::indexes::EmbedderIndex;
//!
//! let registry = Arc::new(EmbedderIndexRegistry::new());
//! let search = MatrixStrategySearch::new(registry);
//!
//! let mut queries = HashMap::new();
//! queries.insert(EmbedderIndex::E1Semantic, vec![0.5f32; 1024]);
//! queries.insert(EmbedderIndex::E7Code, vec![0.5f32; 1536]);
//!
//! // Search with predefined matrix
//! let results = search.search(
//!     queries,
//!     SearchMatrix::code_heavy(),
//!     10,
//!     None,
//! );
//! ```

mod analysis;
mod builder;
mod results;
mod search_matrix;
mod strategy_search;

#[cfg(test)]
mod tests_boundary;
#[cfg(test)]
mod tests_unit;

// Re-export all public items for backwards compatibility
pub use analysis::{CorrelationAnalysis, CorrelationPattern, MatrixAnalysis};
pub use builder::MatrixSearchBuilder;
pub use results::MatrixSearchResults;
pub use search_matrix::SearchMatrix;
pub use strategy_search::MatrixStrategySearch;
