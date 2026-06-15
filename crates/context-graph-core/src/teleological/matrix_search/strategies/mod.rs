//! Search strategy implementations for teleological matrix search.
//!
//! Contains the core similarity computation algorithms:
//! - Cosine similarity
//! - Euclidean distance-based similarity
//! - Synergy-weighted similarity
//! - Group hierarchical similarity
//! - Cross-correlation dominant similarity
//! - Tucker compressed similarity
//! - Adaptive strategy selection

mod computer;
mod helpers;

// Re-export all public items
pub use computer::SimilarityComputer;
pub use helpers::{
    compute_correlation_similarity, compute_purpose_similarity,
    compute_single_embedder_pattern_similarity, compute_specific_groups_similarity,
    compute_specific_pairs_similarity,
};
