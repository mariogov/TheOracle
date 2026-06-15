//! SimilarityComputer implementation for strategy-based comparisons.
//!
//! Contains the core similarity computation algorithms based on configured strategy.

use super::super::super::groups::GroupType;
use super::super::super::synergy_matrix::SynergyMatrix;
use super::super::super::vector::TeleologicalVector;
use super::super::config::MatrixSearchConfig;
use super::super::types::{ComparisonScope, SearchStrategy};
use super::helpers::{
    compute_correlation_similarity, compute_purpose_similarity,
    compute_single_embedder_pattern_similarity, compute_specific_groups_similarity,
    compute_specific_pairs_similarity,
};

/// Core similarity computation methods.
///
/// These functions are used by `TeleologicalMatrixSearch` to compute
/// similarity scores between teleological vectors.
pub struct SimilarityComputer<'a> {
    config: &'a MatrixSearchConfig,
}

impl<'a> SimilarityComputer<'a> {
    /// Create a new similarity computer with the given configuration.
    pub fn new(config: &'a MatrixSearchConfig) -> Self {
        Self { config }
    }

    /// Compute similarity based on the configured strategy.
    pub fn compute(&self, a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
        match self.config.strategy {
            SearchStrategy::Cosine => self.cosine_similarity(a, b),
            SearchStrategy::Euclidean => self.euclidean_similarity(a, b),
            SearchStrategy::SynergyWeighted => self.synergy_weighted_similarity(a, b),
            SearchStrategy::GroupHierarchical => self.group_hierarchical_similarity(a, b),
            SearchStrategy::CrossCorrelationDominant => self.cross_correlation_similarity(a, b),
            SearchStrategy::TuckerCompressed => self.tucker_similarity(a, b),
            SearchStrategy::Adaptive => self.adaptive_similarity(a, b),
        }
    }

    /// Cosine similarity across all components based on scope.
    pub fn cosine_similarity(&self, a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
        match &self.config.scope {
            ComparisonScope::Full => {
                let w = &self.config.weights;
                let pv_sim = compute_purpose_similarity(a, b);
                let cc_sim = compute_correlation_similarity(a, b);
                let ga_sim = a.group_alignments.similarity(&b.group_alignments);
                let conf_sim = a.confidence.min(b.confidence);

                w.topic_profile * pv_sim
                    + w.cross_correlations * cc_sim
                    + w.group_alignments * ga_sim
                    + w.confidence * conf_sim
            }
            ComparisonScope::TopicProfileOnly => compute_purpose_similarity(a, b),
            ComparisonScope::CrossCorrelationsOnly => compute_correlation_similarity(a, b),
            ComparisonScope::GroupAlignmentsOnly => {
                a.group_alignments.similarity(&b.group_alignments)
            }
            ComparisonScope::SpecificPairs(pairs) => compute_specific_pairs_similarity(a, b, pairs),
            ComparisonScope::SpecificGroups(groups) => {
                compute_specific_groups_similarity(a, b, groups)
            }
            ComparisonScope::SingleEmbedderPattern(embedder_idx) => {
                compute_single_embedder_pattern_similarity(a, b, *embedder_idx)
            }
        }
    }

    /// Euclidean distance converted to similarity [0, 1].
    pub fn euclidean_similarity(&self, a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
        let mut sum_sq = 0.0f32;

        // Topic profile distance
        for (&av, &bv) in a
            .topic_profile
            .alignments
            .iter()
            .zip(b.topic_profile.alignments.iter())
        {
            let diff = av - bv;
            sum_sq += diff * diff;
        }

        // Cross-correlation distance
        for (&av, &bv) in a.cross_correlations.iter().zip(b.cross_correlations.iter()) {
            let diff = av - bv;
            sum_sq += diff * diff;
        }

        // Group alignment distance
        let ga = a.group_alignments.as_array();
        let gb = b.group_alignments.as_array();
        for (&av, &bv) in ga.iter().zip(gb.iter()) {
            let diff = av - bv;
            sum_sq += diff * diff;
        }

        // Convert distance to similarity: 1 / (1 + sqrt(distance))
        1.0 / (1.0 + sum_sq.sqrt())
    }

    /// Synergy-weighted similarity using the synergy matrix.
    pub fn synergy_weighted_similarity(
        &self,
        a: &TeleologicalVector,
        b: &TeleologicalVector,
    ) -> f32 {
        let synergy = match &self.config.synergy_matrix {
            Some(s) => s,
            None => return self.cosine_similarity(a, b), // Fall back to cosine
        };

        // Weight each cross-correlation by its synergy value
        let mut weighted_dot = 0.0f32;
        let mut weighted_norm_a = 0.0f32;
        let mut weighted_norm_b = 0.0f32;

        for (flat_idx, (&av, &bv)) in a
            .cross_correlations
            .iter()
            .zip(b.cross_correlations.iter())
            .enumerate()
        {
            let (i, j) = SynergyMatrix::flat_to_indices(flat_idx);
            let weight = synergy.get_weighted_synergy(i, j);

            weighted_dot += weight * av * bv;
            weighted_norm_a += weight * av * av;
            weighted_norm_b += weight * bv * bv;
        }

        let corr_sim = if weighted_norm_a > f32::EPSILON && weighted_norm_b > f32::EPSILON {
            weighted_dot / (weighted_norm_a.sqrt() * weighted_norm_b.sqrt())
        } else {
            0.0
        };

        // Combine with topic profile similarity
        let pv_sim = compute_purpose_similarity(a, b);

        0.4 * pv_sim + 0.6 * corr_sim
    }

    /// Group-hierarchical similarity (compare at group level).
    pub fn group_hierarchical_similarity(
        &self,
        a: &TeleologicalVector,
        b: &TeleologicalVector,
    ) -> f32 {
        // First compute group-level similarity
        let group_sim = a.group_alignments.similarity(&b.group_alignments);

        // Then compare within-group correlation patterns
        let mut within_group_sim = 0.0f32;
        let mut group_count = 0;

        for group in GroupType::ALL {
            let indices = group.embedding_indices();
            if indices.len() < 2 {
                continue;
            }

            // For each pair within the group
            let mut group_corr_sim = 0.0f32;
            let mut pair_count = 0;

            for (k, &idx_k) in indices.iter().enumerate() {
                for &idx_l in indices.iter().skip(k + 1) {
                    let (lo, hi) = if idx_k < idx_l {
                        (idx_k, idx_l)
                    } else {
                        (idx_l, idx_k)
                    };

                    let av = a.get_correlation(lo, hi);
                    let bv = b.get_correlation(lo, hi);

                    // Product similarity
                    group_corr_sim += 1.0 - (av - bv).abs();
                    pair_count += 1;
                }
            }

            if pair_count > 0 {
                within_group_sim += group_corr_sim / pair_count as f32;
                group_count += 1;
            }
        }

        let within_sim = if group_count > 0 {
            within_group_sim / group_count as f32
        } else {
            1.0
        };

        0.6 * group_sim + 0.4 * within_sim
    }

    /// Cross-correlation dominant similarity (prioritize 78 pairs).
    pub fn cross_correlation_similarity(
        &self,
        a: &TeleologicalVector,
        b: &TeleologicalVector,
    ) -> f32 {
        let corr_sim = compute_correlation_similarity(a, b);
        let pv_sim = compute_purpose_similarity(a, b);

        // Heavy weight on correlations
        0.75 * corr_sim + 0.25 * pv_sim
    }

    /// Tucker compressed similarity (if available).
    pub fn tucker_similarity(&self, a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
        match (&a.tucker_core, &b.tucker_core) {
            (Some(ta), Some(tb)) => {
                // Compare Tucker core tensors
                if ta.ranks != tb.ranks {
                    // Different ranks - fall back to regular similarity
                    return self.cosine_similarity(a, b);
                }

                let mut dot = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;

                for (&av, &bv) in ta.data.iter().zip(tb.data.iter()) {
                    dot += av * bv;
                    norm_a += av * av;
                    norm_b += bv * bv;
                }

                if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
                    dot / (norm_a.sqrt() * norm_b.sqrt())
                } else {
                    0.0
                }
            }
            _ => self.cosine_similarity(a, b), // Fall back if Tucker not available
        }
    }

    /// Adaptive similarity: choose best strategy based on vector characteristics.
    pub fn adaptive_similarity(&self, a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
        // Analyze vector characteristics
        let a_density = a.correlation_density();
        let b_density = b.correlation_density();
        let avg_density = (a_density + b_density) / 2.0;

        let a_coherence = a.group_alignments.coherence();
        let b_coherence = b.group_alignments.coherence();
        let avg_coherence = (a_coherence + b_coherence) / 2.0;

        // Choose strategy based on characteristics
        if a.has_tucker_core() && b.has_tucker_core() {
            // Use Tucker if available
            self.tucker_similarity(a, b)
        } else if avg_density < 0.3 {
            // Sparse correlations - use topic profile
            compute_purpose_similarity(a, b)
        } else if avg_coherence > 0.8 {
            // High coherence - use group hierarchical
            self.group_hierarchical_similarity(a, b)
        } else {
            // Default: full weighted similarity
            self.cosine_similarity(a, b)
        }
    }
}
