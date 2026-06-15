//! Core TeleologicalMatrixSearch implementation.
//!
//! Contains the main search struct and fundamental similarity methods.

use super::super::super::groups::GroupType;
use super::super::super::synergy_matrix::{SynergyMatrix, CROSS_CORRELATION_COUNT};
use super::super::super::vector::TeleologicalVector;
use super::super::config::MatrixSearchConfig;
use super::super::strategies::{
    compute_correlation_similarity, compute_purpose_similarity, SimilarityComputer,
};
use super::super::types::SimilarityBreakdown;

/// TeleologicalMatrixSearch: The super search algorithm for 13-embedder teleological vectors.
///
/// Enables cross-correlation search across all 13 embedders at multiple levels,
/// with configurable comparison strategies and component weighting.
pub struct TeleologicalMatrixSearch {
    pub(crate) config: MatrixSearchConfig,
}

impl TeleologicalMatrixSearch {
    /// Create a new matrix search with default configuration.
    pub fn new() -> Self {
        Self {
            config: MatrixSearchConfig::default(),
        }
    }

    /// Create with specific configuration.
    pub fn with_config(config: MatrixSearchConfig) -> Self {
        Self { config }
    }

    /// Get current configuration.
    pub fn config(&self) -> &MatrixSearchConfig {
        &self.config
    }

    /// Set configuration.
    pub fn set_config(&mut self, config: MatrixSearchConfig) {
        self.config = config;
    }

    /// Compute similarity between two teleological vectors.
    ///
    /// Returns similarity score in [0, 1] where 1 = identical.
    pub fn similarity(&self, a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
        let computer = SimilarityComputer::new(&self.config);
        computer.compute(a, b)
    }

    /// Compute similarity with full breakdown.
    pub fn similarity_with_breakdown(
        &self,
        a: &TeleologicalVector,
        b: &TeleologicalVector,
    ) -> SimilarityBreakdown {
        let mut breakdown = SimilarityBreakdown {
            strategy_used: self.config.strategy,
            ..Default::default()
        };

        // Topic profile similarity
        breakdown.topic_profile = compute_purpose_similarity(a, b);

        // Per-embedder topic similarity
        for ((out, &av), &bv) in breakdown
            .per_embedder_topic
            .iter_mut()
            .zip(a.topic_profile.alignments.iter())
            .zip(b.topic_profile.alignments.iter())
        {
            // Product similarity for aligned values
            *out = if av.signum() == bv.signum() {
                1.0 - (av - bv).abs() / 2.0
            } else {
                0.0
            };
        }

        // Cross-correlation similarity
        breakdown.cross_correlations = compute_correlation_similarity(a, b);

        // Find top contributing pairs
        let mut pairs_with_sim: Vec<((usize, usize), f32)> =
            Vec::with_capacity(CROSS_CORRELATION_COUNT);
        for (flat_idx, (&av, &bv)) in a
            .cross_correlations
            .iter()
            .zip(b.cross_correlations.iter())
            .enumerate()
        {
            let (i, j) = SynergyMatrix::flat_to_indices(flat_idx);
            // Contribution = product (high if both agree)
            let contrib = av * bv;
            pairs_with_sim.push(((i, j), contrib));
        }
        pairs_with_sim.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        breakdown.top_correlation_pairs = pairs_with_sim.into_iter().take(10).collect();

        // Group alignments similarity
        breakdown.group_alignments = a.group_alignments.similarity(&b.group_alignments);

        // Per-group similarity
        for group in GroupType::ALL {
            let ga = a.group_alignments.get(group);
            let gb = b.group_alignments.get(group);
            let sim = 1.0 - (ga - gb).abs();
            breakdown.per_group.insert(group, sim);
        }

        // Overall based on weights
        let w = &self.config.weights;
        breakdown.overall = w.topic_profile * breakdown.topic_profile
            + w.cross_correlations * breakdown.cross_correlations
            + w.group_alignments * breakdown.group_alignments
            + w.confidence * (a.confidence.min(b.confidence));

        breakdown
    }

    /// Search for similar vectors in a collection.
    ///
    /// Returns vector of (index, similarity) sorted by descending similarity.
    pub fn search(
        &self,
        query: &TeleologicalVector,
        candidates: &[TeleologicalVector],
    ) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| (idx, self.similarity(query, candidate)))
            .filter(|(_, sim)| *sim >= self.config.min_similarity)
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit results
        results.truncate(self.config.max_results);

        results
    }

    /// Search with full breakdown for each result.
    pub fn search_with_breakdown(
        &self,
        query: &TeleologicalVector,
        candidates: &[TeleologicalVector],
    ) -> Vec<(usize, SimilarityBreakdown)> {
        let mut results: Vec<(usize, SimilarityBreakdown)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| (idx, self.similarity_with_breakdown(query, candidate)))
            .filter(|(_, breakdown)| breakdown.overall >= self.config.min_similarity)
            .collect();

        // Sort by overall similarity descending
        results.sort_by(|a, b| {
            b.1.overall
                .partial_cmp(&a.1.overall)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(self.config.max_results);

        results
    }
}

impl Default for TeleologicalMatrixSearch {
    fn default() -> Self {
        Self::new()
    }
}
