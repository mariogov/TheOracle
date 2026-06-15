//! TeleologicalMatrixSearch: Cross-correlation search across all 13 embedders.
//!
//! This module implements the "super search algorithm" for teleological vectors,
//! enabling multi-level cosine similarity comparisons across the full 13x13
//! embedding matrix.
//!
//! # Search Levels
//!
//! 1. **Full Matrix (13x13)**: Compare all 78 cross-correlations
//! 2. **Topic Profile (13D)**: Compare per-embedder topic alignments
//! 3. **Group Level (6D)**: Compare 6 hierarchical group alignments
//! 4. **Single Embedder**: Compare specific embedder correlation patterns
//! 5. **Synergy-Weighted**: Use learned synergy matrix as similarity weights
//!
//! # Comparison Strategies
//!
//! - `Cosine`: Standard cosine similarity (normalized dot product)
//! - `Euclidean`: L2 distance (inverted to similarity)
//! - `SynergyWeighted`: Synergy matrix modulates importance
//! - `GroupHierarchical`: Aggregate by embedding groups
//! - `CrossCorrelationDominant`: Prioritize 78 pair interactions
//! - `TuckerCompressed`: Use Tucker decomposition for compressed comparison

mod config;
pub mod embedder_names;
mod search;
mod strategies;
mod types;
mod weights;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use config::MatrixSearchConfig;
pub use search::TeleologicalMatrixSearch;
pub use strategies::{
    compute_correlation_similarity, compute_purpose_similarity,
    compute_single_embedder_pattern_similarity, compute_specific_groups_similarity,
    compute_specific_pairs_similarity, SimilarityComputer,
};
pub use types::{ComparisonScope, ComprehensiveComparison, SearchStrategy, SimilarityBreakdown};
pub use weights::ComponentWeights;
