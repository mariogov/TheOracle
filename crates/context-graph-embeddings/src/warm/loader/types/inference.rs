//! Inference Validation types (TASK-EMB-016).
//!
//! # Constitution Alignment
//!
//! - AP-007: Output MUST NOT be sin wave or all zeros
//! - Validates model produces meaningful real output

use std::time::Duration;

/// Inference validation result with golden reference comparison.
///
/// # Constitution Alignment
///
/// - AP-007: Output MUST NOT be sin wave or all zeros
/// - Validates model produces meaningful real output
///
/// # CRITICAL: No Simulation
///
/// The `is_real()` method detects fake inference patterns:
/// - Sin wave patterns: `(i * 0.001).sin()`
/// - All-zero outputs
/// - Low golden similarity (<0.95)
///
/// # Example
///
/// ```rust,ignore
/// use std::time::Duration;
/// use context_graph_embeddings::warm::loader::types::InferenceValidation;
///
/// // Real inference output (not a sin wave, not zeros)
/// let output: Vec<f32> = (0..768)
///     .map(|i| ((i * 17 + 42) % 1000) as f32 / 1000.0 - 0.5)
///     .collect();
///
/// let validation = InferenceValidation::new(
///     "The quick brown fox".to_string(),
///     output,
///     1.0,
///     Duration::from_millis(50),
///     true,
///     0.98,  // High golden similarity
/// );
///
/// assert!(validation.is_real());
/// validation.assert_real(); // Panics on fake patterns
/// ```
#[derive(Debug, Clone)]
pub struct InferenceValidation {
    /// Sample input used for validation (e.g., "The quick brown fox").
    ///
    /// # Invariant
    /// MUST NOT be empty. Real inference requires input.
    pub sample_input: String,

    /// Sample output (embedding vector).
    ///
    /// # Invariant
    /// MUST NOT be empty. MUST NOT be sin wave pattern or all zeros.
    pub sample_output: Vec<f32>,

    /// L2 norm of output (should be ~1.0 for normalized embeddings).
    pub output_norm: f32,

    /// Inference latency.
    pub latency: Duration,

    /// Whether output matches golden reference within tolerance.
    pub matches_golden: bool,

    /// Cosine similarity to golden reference (0.0 to 1.0).
    ///
    /// Must be > 0.95 for real inference to pass `is_real()`.
    pub golden_similarity: f32,
}

impl InferenceValidation {
    /// Minimum golden similarity required for `is_real()` to pass.
    pub const MIN_GOLDEN_SIMILARITY: f32 = 0.95;

    /// Maximum variance for sin wave detection (suspiciously smooth).
    pub const SIN_WAVE_VARIANCE_THRESHOLD: f32 = 0.0001;

    /// Minimum absolute value sum for non-zero detection.
    pub const ZERO_THRESHOLD: f32 = 1e-6;

    /// Create new InferenceValidation with fail-fast validation.
    ///
    /// # Arguments
    ///
    /// * `sample_input` - Text input used for validation (must be non-empty)
    /// * `sample_output` - Embedding output vector (must be non-empty)
    /// * `output_norm` - L2 norm of the output
    /// * `latency` - Inference duration
    /// * `matches_golden` - Whether output matches golden reference
    /// * `golden_similarity` - Cosine similarity to golden reference
    ///
    /// # Panics
    ///
    /// - If `sample_input` is empty
    /// - If `sample_output` is empty
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data.
    #[must_use]
    pub fn new(
        sample_input: String,
        sample_output: Vec<f32>,
        output_norm: f32,
        latency: Duration,
        matches_golden: bool,
        golden_similarity: f32,
    ) -> Self {
        assert!(
            !sample_input.is_empty(),
            "CONSTITUTION VIOLATION AP-007: sample_input is empty. \
             Real test input required."
        );
        assert!(
            !sample_output.is_empty(),
            "CONSTITUTION VIOLATION AP-007: sample_output is empty. \
             Real inference output required."
        );

        Self {
            sample_input,
            sample_output,
            output_norm,
            latency,
            matches_golden,
            golden_similarity,
        }
    }

    /// Check if output looks like real inference (not fake pattern).
    ///
    /// Detects:
    /// - Sin wave patterns (suspiciously smooth consecutive differences)
    /// - All-zero outputs
    /// - Low golden similarity (<0.95)
    ///
    /// # Returns
    ///
    /// `true` if output appears to be from real GPU inference.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Real output with varied values
    /// let real_output: Vec<f32> = (0..768)
    ///     .map(|i| ((i * 17 + 42) % 1000) as f32 / 1000.0 - 0.5)
    ///     .collect();
    /// let validation = InferenceValidation::new(
    ///     "test".to_string(), real_output, 1.0, Duration::from_millis(10), true, 0.99
    /// );
    /// assert!(validation.is_real());
    ///
    /// // Fake: sin wave pattern
    /// let sin_wave: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001).sin()).collect();
    /// let fake = InferenceValidation::new(
    ///     "test".to_string(), sin_wave, 1.0, Duration::from_millis(10), true, 0.99
    /// );
    /// assert!(!fake.is_real()); // Detected as sin wave
    /// ```
    #[must_use]
    pub fn is_real(&self) -> bool {
        // Check 1: All zeros detection
        let is_zeros = self
            .sample_output
            .iter()
            .all(|&v| v.abs() < Self::ZERO_THRESHOLD);
        if is_zeros {
            return false;
        }

        // Check 2: Sin wave pattern detection
        // A sin wave has suspiciously smooth consecutive differences
        if self.detect_sin_wave_pattern() {
            return false;
        }

        // Check 3: Golden similarity must be high for real model
        if self.golden_similarity < Self::MIN_GOLDEN_SIMILARITY {
            return false;
        }

        true
    }

    /// Detect sin wave fake pattern.
    ///
    /// Sin waves have very low variance in their consecutive differences
    /// because the derivative of sin is cos, which is also smooth.
    fn detect_sin_wave_pattern(&self) -> bool {
        if self.sample_output.len() < 10 {
            return false;
        }

        // Check all windows of 10 elements for suspiciously smooth differences
        self.sample_output.windows(10).all(|w| {
            // Calculate consecutive differences
            let diffs: Vec<f32> = w.windows(2).map(|p| (p[1] - p[0]).abs()).collect();

            // Calculate variance of differences
            let mean: f32 = diffs.iter().sum::<f32>() / diffs.len() as f32;
            let variance: f32 =
                diffs.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32;

            // Suspiciously smooth if variance is too low
            variance < Self::SIN_WAVE_VARIANCE_THRESHOLD
        })
    }

    /// Panic if output looks fake.
    ///
    /// # Panics
    ///
    /// Constitution AP-007 violation with error code EMB-E011 if `is_real()` returns false.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Real inference - OK
    /// let real = InferenceValidation::new(...);
    /// real.assert_real();
    ///
    /// // Fake sin wave - PANIC!
    /// let fake_sin = InferenceValidation { sample_output: sin_wave, .. };
    /// fake_sin.assert_real(); // PANIC: "[EMB-E011] FAKE_INFERENCE: ..."
    /// ```
    pub fn assert_real(&self) {
        if !self.is_real() {
            panic!(
                "[EMB-E011] FAKE_INFERENCE: Output pattern indicates simulation. \
                 Golden similarity: {:.4}, output_len: {}. Constitution AP-007 violation.",
                self.golden_similarity,
                self.sample_output.len()
            );
        }
    }

    /// Calculate L2 norm of sample_output for verification.
    ///
    /// # Returns
    ///
    /// L2 norm (Euclidean length) of the output vector.
    #[must_use]
    pub fn calculate_norm(&self) -> f32 {
        self.sample_output.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Verify that stored output_norm matches calculated norm.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed difference between stored and calculated norm
    ///
    /// # Returns
    ///
    /// `true` if stored norm matches calculated norm within tolerance.
    #[must_use]
    pub fn verify_norm(&self, tolerance: f32) -> bool {
        let calculated = self.calculate_norm();
        (self.output_norm - calculated).abs() < tolerance
    }

    /// Get output dimension.
    #[must_use]
    pub fn output_dimension(&self) -> usize {
        self.sample_output.len()
    }
}
