//! Validation methods for SynergyMatrix.

use crate::teleological::comparison_error::{
    ComparisonValidationError, ComparisonValidationResult,
};

use super::constants::SYNERGY_DIM;
use super::types::SynergyMatrix;

impl SynergyMatrix {
    /// Default tolerance for matrix validation
    pub const VALIDATION_TOLERANCE: f32 = 1e-6;

    /// Check if the matrix is symmetric within tolerance.
    pub fn is_symmetric(&self, tolerance: f32) -> bool {
        for (i, row) in self.values.iter().enumerate() {
            for (j, &value) in row.iter().enumerate().skip(i + 1) {
                if (value - self.values[j][i]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Check if diagonal values are all 1.0 within tolerance.
    pub fn has_unit_diagonal(&self, tolerance: f32) -> bool {
        for (i, row) in self.values.iter().enumerate() {
            if (row[i] - 1.0).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Check if all values are in [0.0, 1.0].
    pub fn values_in_range(&self) -> bool {
        for row in &self.values {
            for &value in row {
                if !(0.0..=1.0).contains(&value) {
                    return false;
                }
            }
        }
        true
    }

    /// Validate all matrix invariants.
    ///
    /// Returns `Ok(())` if:
    /// - Matrix is symmetric (within tolerance)
    /// - All diagonal values are 1.0 (within tolerance)
    /// - All values are in [0.0, 1.0]
    ///
    /// Returns detailed error describing exactly what failed.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::teleological::SynergyMatrix;
    ///
    /// let valid = SynergyMatrix::with_base_synergies();
    /// assert!(valid.validate().is_ok());
    /// ```
    pub fn validate(&self) -> ComparisonValidationResult<()> {
        self.validate_with_tolerance(Self::VALIDATION_TOLERANCE)
    }

    /// Validate matrix invariants with custom tolerance.
    pub fn validate_with_tolerance(&self, tolerance: f32) -> ComparisonValidationResult<()> {
        // Check symmetry
        for i in 0..SYNERGY_DIM {
            for j in (i + 1)..SYNERGY_DIM {
                let diff = (self.values[i][j] - self.values[j][i]).abs();
                if diff > tolerance {
                    return Err(ComparisonValidationError::MatrixNotSymmetric {
                        row: i,
                        col: j,
                        value_ij: self.values[i][j],
                        value_ji: self.values[j][i],
                        tolerance,
                    });
                }
            }
        }

        // Check diagonal is 1.0
        for i in 0..SYNERGY_DIM {
            if (self.values[i][i] - 1.0).abs() > tolerance {
                return Err(ComparisonValidationError::DiagonalNotUnity {
                    index: i,
                    actual: self.values[i][i],
                    expected: 1.0,
                    tolerance,
                });
            }
        }

        // Check values in range
        for i in 0..SYNERGY_DIM {
            for j in 0..SYNERGY_DIM {
                let value = self.values[i][j];
                if !(0.0..=1.0).contains(&value) {
                    return Err(ComparisonValidationError::SynergyOutOfRange {
                        row: i,
                        col: j,
                        value,
                        min: 0.0,
                        max: 1.0,
                    });
                }
            }
        }

        Ok(())
    }

    /// Check if matrix is valid (returns bool for simple checks).
    ///
    /// For detailed error information, use `validate()` instead.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Assert matrix is valid, panicking with detailed error on failure.
    ///
    /// Use this for cases where validation failure is a programmer error.
    pub fn assert_valid(&self) {
        if let Err(e) = self.validate() {
            panic!("FAIL FAST: SynergyMatrix validation failed: {}", e);
        }
    }
}
