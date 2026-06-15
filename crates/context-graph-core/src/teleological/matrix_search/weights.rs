//! Component weights for teleological similarity calculations.
//!
//! Defines how different components of teleological vectors
//! are weighted during comparison.

use serde::{Deserialize, Serialize};

use super::super::comparison_error::{
    ComparisonValidationError, ComparisonValidationResult, WeightValues,
};

/// Weights for different comparison components in Full scope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentWeights {
    /// Weight for topic profile similarity (default 0.4)
    pub topic_profile: f32,
    /// Weight for cross-correlation similarity (default 0.35)
    pub cross_correlations: f32,
    /// Weight for group alignments similarity (default 0.15)
    pub group_alignments: f32,
    /// Weight for confidence factor (default 0.1)
    pub confidence: f32,
}

impl Default for ComponentWeights {
    fn default() -> Self {
        Self {
            topic_profile: 0.4,
            cross_correlations: 0.35,
            group_alignments: 0.15,
            confidence: 0.1,
        }
    }
}

impl ComponentWeights {
    /// Weights emphasizing cross-correlations (for teleological search)
    pub fn correlation_focused() -> Self {
        Self {
            topic_profile: 0.25,
            cross_correlations: 0.55,
            group_alignments: 0.15,
            confidence: 0.05,
        }
    }

    /// Weights emphasizing topic profile (for semantic topic search)
    pub fn topic_focused() -> Self {
        Self {
            topic_profile: 0.6,
            cross_correlations: 0.2,
            group_alignments: 0.15,
            confidence: 0.05,
        }
    }

    /// Weights emphasizing group structure (for hierarchical search)
    pub fn group_focused() -> Self {
        Self {
            topic_profile: 0.25,
            cross_correlations: 0.25,
            group_alignments: 0.45,
            confidence: 0.05,
        }
    }

    /// Tolerance for weight sum validation (0.1%)
    pub const WEIGHT_SUM_TOLERANCE: f32 = 0.001;

    /// Validate all weight invariants.
    ///
    /// Returns `Ok(())` if:
    /// - All weights are in [0.0, 1.0]
    /// - Weights sum to 1.0 (±tolerance)
    ///
    /// Returns detailed error describing exactly what failed.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::teleological::ComponentWeights;
    ///
    /// let valid = ComponentWeights::default();
    /// assert!(valid.validate().is_ok());
    ///
    /// let mut invalid = ComponentWeights::default();
    /// invalid.topic_profile = 2.0; // Out of range
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> ComparisonValidationResult<()> {
        // Check individual weight ranges
        if !(0.0..=1.0).contains(&self.topic_profile) {
            return Err(ComparisonValidationError::WeightOutOfRange {
                field_name: "topic_profile",
                value: self.topic_profile,
                min: 0.0,
                max: 1.0,
            });
        }

        if !(0.0..=1.0).contains(&self.cross_correlations) {
            return Err(ComparisonValidationError::WeightOutOfRange {
                field_name: "cross_correlations",
                value: self.cross_correlations,
                min: 0.0,
                max: 1.0,
            });
        }

        if !(0.0..=1.0).contains(&self.group_alignments) {
            return Err(ComparisonValidationError::WeightOutOfRange {
                field_name: "group_alignments",
                value: self.group_alignments,
                min: 0.0,
                max: 1.0,
            });
        }

        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(ComparisonValidationError::WeightOutOfRange {
                field_name: "confidence",
                value: self.confidence,
                min: 0.0,
                max: 1.0,
            });
        }

        // Check sum
        let sum =
            self.topic_profile + self.cross_correlations + self.group_alignments + self.confidence;

        if (sum - 1.0).abs() > Self::WEIGHT_SUM_TOLERANCE {
            return Err(ComparisonValidationError::WeightsNotNormalized {
                actual_sum: sum,
                expected_sum: 1.0,
                tolerance: Self::WEIGHT_SUM_TOLERANCE,
                weights: WeightValues {
                    topic_profile: self.topic_profile,
                    cross_correlations: self.cross_correlations,
                    group_alignments: self.group_alignments,
                    confidence: self.confidence,
                },
            });
        }

        Ok(())
    }

    /// Check if weights are valid (returns bool for simple checks).
    ///
    /// For detailed error information, use `validate()` instead.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum =
            self.topic_profile + self.cross_correlations + self.group_alignments + self.confidence;
        if sum > f32::EPSILON {
            self.topic_profile /= sum;
            self.cross_correlations /= sum;
            self.group_alignments /= sum;
            self.confidence /= sum;
        }
    }
}
