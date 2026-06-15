//! Accessor and mutator methods for SynergyMatrix.

use chrono::Utc;

use super::constants::SYNERGY_DIM;
use super::types::SynergyMatrix;

impl SynergyMatrix {
    /// Get synergy value between embeddings i and j.
    ///
    /// # Panics
    ///
    /// Panics if `i >= SYNERGY_DIM` or `j >= SYNERGY_DIM` (FAIL FAST).
    #[inline]
    pub fn get_synergy(&self, i: usize, j: usize) -> f32 {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: synergy index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: synergy index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        self.values[i][j]
    }

    /// Set synergy value between embeddings i and j.
    ///
    /// Automatically maintains symmetry (sets both [i][j] and [j][i]).
    ///
    /// # Panics
    ///
    /// - Panics if `i >= SYNERGY_DIM` or `j >= SYNERGY_DIM` (FAIL FAST)
    /// - Panics if `value < 0.0` or `value > 1.0` (FAIL FAST)
    /// - Panics if attempting to set diagonal to non-1.0 value (FAIL FAST)
    #[inline]
    pub fn set_synergy(&mut self, i: usize, j: usize, value: f32) {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: synergy index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: synergy index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        assert!(
            (0.0..=1.0).contains(&value),
            "FAIL FAST: synergy value {} must be in [0.0, 1.0]",
            value
        );
        if i == j {
            assert!(
                (value - 1.0).abs() < f32::EPSILON,
                "FAIL FAST: diagonal synergy must be 1.0, got {}",
                value
            );
        }

        self.values[i][j] = value;
        self.values[j][i] = value; // Maintain symmetry
        self.computed_at = Utc::now();
    }

    /// Get weight for synergy between embeddings i and j.
    ///
    /// # Panics
    ///
    /// Panics if `i >= SYNERGY_DIM` or `j >= SYNERGY_DIM` (FAIL FAST).
    #[inline]
    pub fn get_weight(&self, i: usize, j: usize) -> f32 {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: weight index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: weight index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        self.weights[i][j]
    }

    /// Set weight for synergy between embeddings i and j.
    ///
    /// Automatically maintains symmetry.
    ///
    /// # Panics
    ///
    /// - Panics if indices out of bounds (FAIL FAST)
    /// - Panics if weight is negative (FAIL FAST)
    #[inline]
    pub fn set_weight(&mut self, i: usize, j: usize, weight: f32) {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: weight index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: weight index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        assert!(
            weight >= 0.0,
            "FAIL FAST: weight {} must be non-negative",
            weight
        );

        self.weights[i][j] = weight;
        self.weights[j][i] = weight; // Maintain symmetry
    }

    /// Get weighted synergy value (value * weight).
    #[inline]
    pub fn get_weighted_synergy(&self, i: usize, j: usize) -> f32 {
        self.get_synergy(i, j) * self.get_weight(i, j)
    }
}
