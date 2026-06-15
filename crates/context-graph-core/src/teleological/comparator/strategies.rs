//! Aggregation strategy implementations for teleological comparison.
//!
//! Provides various methods to aggregate per-embedder similarity scores
//! into an overall similarity value based on the search strategy.

use crate::teleological::{MatrixSearchConfig, SearchStrategy, NUM_EMBEDDERS};

/// Trait for aggregation strategy operations.
///
/// This trait is implemented by types that need to aggregate
/// per-embedder similarity scores using different strategies.
pub(crate) trait AggregationStrategies {
    /// Get the configuration for accessing synergy matrix.
    fn config(&self) -> &MatrixSearchConfig;

    /// Aggregate per-embedder scores according to strategy.
    fn aggregate(&self, scores: &[Option<f32>; NUM_EMBEDDERS], strategy: SearchStrategy) -> f32 {
        match strategy {
            SearchStrategy::Cosine => aggregate_mean(scores),
            SearchStrategy::Euclidean => aggregate_euclidean(scores),
            SearchStrategy::SynergyWeighted => self.aggregate_synergy(scores),
            SearchStrategy::GroupHierarchical => aggregate_hierarchical(scores),
            SearchStrategy::CrossCorrelationDominant => self.aggregate_correlation(scores),
            SearchStrategy::TuckerCompressed => aggregate_tucker(scores),
            SearchStrategy::Adaptive => self.aggregate_adaptive(scores),
        }
    }

    /// Synergy-weighted aggregation using SynergyMatrix diagonal weights.
    fn aggregate_synergy(&self, scores: &[Option<f32>; NUM_EMBEDDERS]) -> f32 {
        let synergy = match &self.config().synergy_matrix {
            Some(s) => s,
            None => return aggregate_mean(scores), // Fallback if no synergy matrix
        };

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, score) in scores.iter().enumerate() {
            if let Some(s) = score {
                // Diagonal elements represent self-importance weights
                let weight = synergy.get_synergy(i, i);
                weighted_sum += s * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > f32::EPSILON {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Cross-correlation dominant: emphasizes pairs with high synergy.
    fn aggregate_correlation(&self, scores: &[Option<f32>; NUM_EMBEDDERS]) -> f32 {
        let synergy = match &self.config().synergy_matrix {
            Some(s) => s,
            None => return aggregate_mean(scores),
        };

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        // Use off-diagonal synergy values to weight pairs
        for i in 0..NUM_EMBEDDERS {
            for j in (i + 1)..NUM_EMBEDDERS {
                if let (Some(s_i), Some(s_j)) = (scores[i], scores[j]) {
                    let pair_sim = (s_i + s_j) / 2.0;
                    let synergy_weight = synergy.get_synergy(i, j);
                    weighted_sum += pair_sim * synergy_weight;
                    weight_sum += synergy_weight;
                }
            }
        }

        if weight_sum > f32::EPSILON {
            weighted_sum / weight_sum
        } else {
            aggregate_mean(scores)
        }
    }

    /// Adaptive strategy: chooses aggregation based on score distribution.
    fn aggregate_adaptive(&self, scores: &[Option<f32>; NUM_EMBEDDERS]) -> f32 {
        let valid_scores: Vec<f32> = scores.iter().filter_map(|&s| s).collect();
        if valid_scores.is_empty() {
            return 0.0;
        }

        let n = valid_scores.len() as f32;
        let mean: f32 = valid_scores.iter().sum::<f32>() / n;
        let variance: f32 = valid_scores
            .iter()
            .map(|&s| (s - mean).powi(2))
            .sum::<f32>()
            / n;
        let std_dev = variance.sqrt();

        // Coefficient of variation determines strategy
        let cov = if mean > f32::EPSILON {
            std_dev / mean
        } else {
            0.0
        };

        if cov < 0.1 {
            // Low variance: scores are consistent, use simple mean
            aggregate_mean(scores)
        } else if cov < 0.3 {
            // Medium variance: use synergy weighting if available
            if self.config().synergy_matrix.is_some() {
                self.aggregate_synergy(scores)
            } else {
                aggregate_hierarchical(scores)
            }
        } else {
            // High variance: use robust method
            aggregate_tucker(scores)
        }
    }
}

/// Simple weighted mean of available scores.
pub(crate) fn aggregate_mean(scores: &[Option<f32>; NUM_EMBEDDERS]) -> f32 {
    let valid_scores: Vec<f32> = scores.iter().filter_map(|&s| s).collect();
    if valid_scores.is_empty() {
        return 0.0;
    }
    let sum: f32 = valid_scores.iter().sum();
    sum / valid_scores.len() as f32
}

/// Euclidean distance converted to similarity.
/// Uses: similarity = 1 / (1 + sqrt(sum((1-s_i)^2) / n))
pub(crate) fn aggregate_euclidean(scores: &[Option<f32>; NUM_EMBEDDERS]) -> f32 {
    let valid_scores: Vec<f32> = scores.iter().filter_map(|&s| s).collect();
    if valid_scores.is_empty() {
        return 0.0;
    }
    let n = valid_scores.len() as f32;
    let sum_sq: f32 = valid_scores.iter().map(|&s| (1.0 - s).powi(2)).sum();
    let rms_distance = (sum_sq / n).sqrt();
    1.0 / (1.0 + rms_distance)
}

/// Group-hierarchical aggregation using embedding groups.
/// Groups: Factual (E1,E12,E13), Temporal (E2,E3), Causal (E4,E7),
///         Relational (E5,E8,E9), Qualitative (E10,E11), Implementation (E6)
pub(crate) fn aggregate_hierarchical(scores: &[Option<f32>; NUM_EMBEDDERS]) -> f32 {
    // Define group indices
    let groups: [&[usize]; 6] = [
        &[0, 10, 11, 12], // Factual: E1, E11, E12, E13
        &[1, 2, 3],       // Temporal: E2, E3, E4
        &[4, 6],          // Causal: E5, E7
        &[7, 8],          // Relational: E8, E9
        &[9],             // Qualitative: E10
        &[5],             // Implementation: E6
    ];

    // Group weights (equal weighting across groups)
    let group_weight = 1.0 / groups.len() as f32;

    let mut total = 0.0;
    let mut valid_groups = 0;

    for group_indices in groups.iter() {
        let group_scores: Vec<f32> = group_indices.iter().filter_map(|&i| scores[i]).collect();

        if !group_scores.is_empty() {
            let group_mean: f32 = group_scores.iter().sum::<f32>() / group_scores.len() as f32;
            total += group_mean * group_weight;
            valid_groups += 1;
        }
    }

    if valid_groups > 0 {
        // Normalize by actual number of groups with valid scores
        total * (groups.len() as f32 / valid_groups as f32)
    } else {
        0.0
    }
}

/// Tucker decomposition approximation (simplified for fingerprint comparison).
/// Uses principal components estimated from score variance.
pub(crate) fn aggregate_tucker(scores: &[Option<f32>; NUM_EMBEDDERS]) -> f32 {
    let valid_scores: Vec<f32> = scores.iter().filter_map(|&s| s).collect();
    if valid_scores.is_empty() {
        return 0.0;
    }

    let n = valid_scores.len() as f32;
    let mean: f32 = valid_scores.iter().sum::<f32>() / n;
    let variance: f32 = valid_scores
        .iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();

    // Tucker-inspired: weight by how close each score is to the mean
    // (approximates principal component contribution)
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;

    for &score in &valid_scores {
        // Higher weight for scores closer to mean (more "typical")
        let distance = (score - mean).abs();
        let weight = if std_dev > f32::EPSILON {
            (1.0 - (distance / (std_dev * 2.0)).min(1.0)).max(0.1)
        } else {
            1.0
        };
        weighted_sum += score * weight;
        weight_sum += weight;
    }

    if weight_sum > f32::EPSILON {
        weighted_sum / weight_sum
    } else {
        mean
    }
}
