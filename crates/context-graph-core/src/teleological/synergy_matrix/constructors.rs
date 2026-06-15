//! Core constructors for SynergyMatrix.

use chrono::Utc;

use super::constants::{BASE_SYNERGIES, SYNERGY_DIM};
use super::types::SynergyMatrix;

impl SynergyMatrix {
    /// Base synergies from teleoplan.md synergy matrix.
    ///
    /// Values: weak (0.3), moderate (0.6), strong (0.9)
    ///
    /// Index mapping:
    /// - 0: E1_Semantic
    /// - 1: E2_Episodic
    /// - 2: E3_Temporal
    /// - 3: E4_Causal
    /// - 4: E5_Analogical
    /// - 5: E6_Code
    /// - 6: E7_Procedural
    /// - 7: E8_Spatial
    /// - 8: E9_Social
    /// - 9: E10_Emotional
    /// - 10: E11_Abstract
    /// - 11: E12_Factual
    /// - 12: E13_Sparse
    pub const BASE_SYNERGIES: [[f32; SYNERGY_DIM]; SYNERGY_DIM] = BASE_SYNERGIES;

    /// Create a new empty synergy matrix with identity diagonal.
    ///
    /// All synergy values are 0.0 except diagonal which is 1.0.
    pub fn new() -> Self {
        let mut values = [[0.0f32; SYNERGY_DIM]; SYNERGY_DIM];
        let mut weights = [[1.0f32; SYNERGY_DIM]; SYNERGY_DIM];

        // Set diagonal to 1.0
        for (i, (val_row, wgt_row)) in values.iter_mut().zip(weights.iter_mut()).enumerate() {
            val_row[i] = 1.0;
            wgt_row[i] = 1.0;
        }

        Self {
            values,
            weights,
            computed_at: Utc::now(),
            sample_count: 0,
        }
    }

    /// Create a synergy matrix initialized with base synergies from teleoplan.md.
    pub fn with_base_synergies() -> Self {
        Self {
            values: BASE_SYNERGIES,
            weights: [[1.0f32; SYNERGY_DIM]; SYNERGY_DIM],
            computed_at: Utc::now(),
            sample_count: 0,
        }
    }

    /// Create a balanced synergy matrix.
    ///
    /// Uses moderate synergies (0.6) across all pairs for unbiased retrieval.
    ///
    /// Use for: general-purpose search, exploration, discovery.
    pub fn balanced() -> Self {
        let mut values = [[0.6f32; SYNERGY_DIM]; SYNERGY_DIM];

        // Set diagonal to 1.0
        for (i, row) in values.iter_mut().enumerate() {
            row[i] = 1.0;
        }

        Self {
            values,
            weights: [[1.0f32; SYNERGY_DIM]; SYNERGY_DIM],
            computed_at: Utc::now(),
            sample_count: 0,
        }
    }

    /// Create an identity synergy matrix.
    ///
    /// Diagonal is 1.0, all off-diagonal is 0.0 (no cross-embedder synergy).
    ///
    /// Use for: per-embedder independent search, testing, baseline comparison.
    pub fn identity() -> Self {
        Self::new()
    }
}

impl Default for SynergyMatrix {
    fn default() -> Self {
        Self::with_base_synergies()
    }
}
