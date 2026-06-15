//! Analysis and utility methods for SynergyMatrix.

use super::constants::{CROSS_CORRELATION_COUNT, SYNERGY_DIM};
use super::types::SynergyMatrix;

impl SynergyMatrix {
    /// Compute average synergy across all pairs (excluding diagonal).
    pub fn average_synergy(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0;

        for (i, row) in self.values.iter().enumerate() {
            for &value in row.iter().skip(i + 1) {
                sum += value;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Get indices of high synergy pairs (value >= threshold).
    pub fn high_synergy_pairs(&self, threshold: f32) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for (i, row) in self.values.iter().enumerate() {
            for (j, &value) in row.iter().enumerate().skip(i + 1) {
                if value >= threshold {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Flatten upper triangle to cross-correlation array.
    ///
    /// Returns 78 values corresponding to unique pairs (i, j) where i < j.
    /// Order: (0,1), (0,2), ..., (0,12), (1,2), (1,3), ..., (11,12)
    pub fn to_cross_correlations(&self) -> [f32; CROSS_CORRELATION_COUNT] {
        let mut result = [0.0f32; CROSS_CORRELATION_COUNT];
        let mut idx = 0;

        for (i, row) in self.values.iter().enumerate() {
            for &value in row.iter().skip(i + 1) {
                result[idx] = value;
                idx += 1;
            }
        }

        result
    }

    /// Convert flat index (0-77) to matrix indices (i, j).
    ///
    /// # Panics
    ///
    /// Panics if `flat_idx >= CROSS_CORRELATION_COUNT` (FAIL FAST).
    pub fn flat_to_indices(flat_idx: usize) -> (usize, usize) {
        assert!(
            flat_idx < CROSS_CORRELATION_COUNT,
            "FAIL FAST: flat index {} out of bounds (max {})",
            flat_idx,
            CROSS_CORRELATION_COUNT - 1
        );

        // Inverse of triangular number formula
        let mut i = 0;
        let mut offset = 0;

        while offset + (SYNERGY_DIM - 1 - i) <= flat_idx {
            offset += SYNERGY_DIM - 1 - i;
            i += 1;
        }

        let j = flat_idx - offset + i + 1;
        (i, j)
    }

    /// Convert matrix indices (i, j) to flat index (0-77).
    ///
    /// # Panics
    ///
    /// Panics if indices out of bounds or i >= j (FAIL FAST).
    pub fn indices_to_flat(i: usize, j: usize) -> usize {
        assert!(
            i < SYNERGY_DIM && j < SYNERGY_DIM,
            "FAIL FAST: indices ({}, {}) out of bounds",
            i,
            j
        );
        assert!(i < j, "FAIL FAST: i ({}) must be less than j ({})", i, j);

        let mut flat_idx = 0;
        for row in 0..i {
            flat_idx += SYNERGY_DIM - 1 - row;
        }
        flat_idx + (j - i - 1)
    }
}
