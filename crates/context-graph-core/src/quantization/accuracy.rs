//! Accuracy verification for quantization.
//!
//! TASK-L03: Provides RMSE and NRMSE computation for quantization accuracy verification.

use super::types::Precision;

/// Compute Root Mean Square Error between original and reconstructed values.
///
/// # Arguments
///
/// * `original` - Original values
/// * `reconstructed` - Reconstructed values after quantization/dequantization
///
/// # Returns
///
/// RMSE value, or NaN if inputs are invalid.
pub fn compute_rmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.len() != reconstructed.len() || original.is_empty() {
        return f32::NAN;
    }

    let sum_sq: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    (sum_sq / original.len() as f32).sqrt()
}

/// Compute Normalized Root Mean Square Error (relative to value range).
///
/// NRMSE = RMSE / (max - min)
///
/// This is useful for comparing accuracy across different value ranges.
///
/// # Arguments
///
/// * `original` - Original values
/// * `reconstructed` - Reconstructed values
///
/// # Returns
///
/// NRMSE value in [0, 1], or NaN if inputs are invalid.
pub fn compute_nrmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let rmse = compute_rmse(original, reconstructed);
    if rmse.is_nan() {
        return f32::NAN;
    }

    let (min, max) = original
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });

    let range = max - min;
    if range < f32::EPSILON {
        // All values are the same
        if rmse < f32::EPSILON {
            return 0.0; // Perfect reconstruction of constant
        }
        return f32::NAN; // Some error but no range
    }

    rmse / range
}

/// Accuracy verification report.
#[derive(Debug, Clone)]
pub struct AccuracyReport {
    /// Root mean square error
    pub rmse: f32,
    /// Normalized RMSE (relative to value range)
    pub nrmse: f32,
    /// Threshold for this precision level
    pub threshold: f32,
    /// Whether accuracy meets threshold
    pub passed: bool,
    /// Precision level tested
    pub precision: Precision,
    /// Achieved compression ratio
    pub compression_ratio: f32,
}

impl AccuracyReport {
    /// Create a new accuracy report from measurement.
    pub fn new(rmse: f32, nrmse: f32, precision: Precision, compression_ratio: f32) -> Self {
        let threshold = precision.rmse_threshold();
        Self {
            rmse,
            nrmse,
            threshold,
            passed: nrmse <= threshold,
            precision,
            compression_ratio,
        }
    }
}

impl std::fmt::Display for AccuracyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} accuracy: NRMSE={:.4}% (threshold={:.4}%), compression={:.1}x, {}",
            self.precision,
            self.nrmse * 100.0,
            self.threshold * 100.0,
            self.compression_ratio,
            if self.passed { "PASS" } else { "FAIL" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rmse_identical() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rmse = compute_rmse(&values, &values);
        assert!(
            rmse.abs() < f32::EPSILON,
            "RMSE of identical values should be 0"
        );
    }

    #[test]
    fn test_compute_rmse_known_value() {
        let original = vec![0.0, 0.0, 0.0, 0.0];
        let reconstructed = vec![1.0, 1.0, 1.0, 1.0];
        let rmse = compute_rmse(&original, &reconstructed);
        // RMSE of [0,0,0,0] vs [1,1,1,1] = sqrt((4 * 1) / 4) = 1.0
        assert!((rmse - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_rmse_mismatched_length() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let rmse = compute_rmse(&a, &b);
        assert!(rmse.is_nan());
    }

    #[test]
    fn test_compute_rmse_empty() {
        let empty: Vec<f32> = vec![];
        let rmse = compute_rmse(&empty, &empty);
        assert!(rmse.is_nan());
    }

    #[test]
    fn test_compute_nrmse_identical() {
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let nrmse = compute_nrmse(&values, &values);
        assert!(nrmse.abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_nrmse_known_value() {
        let original = vec![0.0, 10.0];
        let reconstructed = vec![1.0, 11.0];
        let nrmse = compute_nrmse(&original, &reconstructed);
        // RMSE = 1.0, range = 10.0, NRMSE = 0.1
        assert!((nrmse - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_compute_nrmse_constant() {
        let values = vec![5.0, 5.0, 5.0];
        let nrmse = compute_nrmse(&values, &values);
        // Constant values with perfect reconstruction
        assert!(nrmse.abs() < f32::EPSILON);
    }

    #[test]
    fn test_accuracy_report_pass() {
        let report = AccuracyReport::new(0.001, 0.005, Precision::Int8, 4.0);
        assert!(report.passed);
        assert!(report.nrmse < report.threshold);
    }

    #[test]
    fn test_accuracy_report_fail() {
        let report = AccuracyReport::new(0.1, 0.1, Precision::Int8, 4.0);
        assert!(!report.passed);
    }

    #[test]
    fn test_accuracy_report_display() {
        let report = AccuracyReport::new(0.001, 0.005, Precision::Int8, 4.0);
        let display = format!("{}", report);
        assert!(display.contains("INT8"));
        assert!(display.contains("PASS"));
    }
}
