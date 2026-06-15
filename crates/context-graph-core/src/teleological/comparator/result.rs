//! ComparisonResult type for teleological fingerprint comparison.

use crate::teleological::{Embedder, SearchStrategy, SimilarityBreakdown, NUM_EMBEDDERS};

/// Result of comparing two teleological fingerprints.
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    /// Overall similarity score [0.0, 1.0]
    pub overall: f32,
    /// Per-embedder similarity scores (None if embedder unavailable or comparison failed)
    pub per_embedder: [Option<f32>; NUM_EMBEDDERS],
    /// Strategy used for comparison
    pub strategy: SearchStrategy,
    /// Coherence: inverse of coefficient of variation across embedders (higher = more consistent)
    pub coherence: Option<f32>,
    /// Dominant embedder (highest score)
    pub dominant_embedder: Option<Embedder>,
    /// Optional detailed breakdown
    pub breakdown: Option<SimilarityBreakdown>,
}

impl ComparisonResult {
    /// Create a new ComparisonResult with default values.
    pub(crate) fn new(strategy: SearchStrategy) -> Self {
        Self {
            overall: 0.0,
            per_embedder: [None; NUM_EMBEDDERS],
            strategy,
            coherence: None,
            dominant_embedder: None,
            breakdown: None,
        }
    }

    /// Count how many embedders have valid scores.
    pub fn valid_score_count(&self) -> usize {
        self.per_embedder.iter().filter(|s| s.is_some()).count()
    }
}
