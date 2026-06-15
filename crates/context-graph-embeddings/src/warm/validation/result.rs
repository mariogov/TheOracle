//! Validation result types.
//!
//! Aggregates results from all validation stages (dimensions, weights,
//! inference). If any stage fails, `error` contains the failure details.

use crate::warm::error::WarmError;

/// Result of model validation after warm loading.
///
/// Aggregates results from all validation stages (dimensions, weights,
/// inference). If any stage fails, `error` contains the failure details.
#[derive(Debug)]
pub struct ValidationResult {
    /// Model identifier that was validated.
    pub model_id: String,
    /// Whether output dimension matches expected.
    pub dimension_valid: bool,
    /// Whether weights contain no NaN/Inf values.
    pub weights_valid: bool,
    /// Whether test inference produced valid output.
    pub inference_valid: bool,
    /// Time taken for test inference in milliseconds.
    pub inference_time_ms: u64,
    /// Error details if any validation stage failed.
    pub error: Option<WarmError>,
}

impl ValidationResult {
    /// Create a successful validation result.
    #[must_use]
    pub fn success(model_id: String, inference_time_ms: u64) -> Self {
        Self {
            model_id,
            dimension_valid: true,
            weights_valid: true,
            inference_valid: true,
            inference_time_ms,
            error: None,
        }
    }

    /// Create a failed validation result.
    #[must_use]
    pub fn failure(
        model_id: String,
        dimension_valid: bool,
        weights_valid: bool,
        inference_valid: bool,
        inference_time_ms: u64,
        error: WarmError,
    ) -> Self {
        Self {
            model_id,
            dimension_valid,
            weights_valid,
            inference_valid,
            inference_time_ms,
            error: Some(error),
        }
    }

    /// Check if all validation stages passed.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.dimension_valid && self.weights_valid && self.inference_valid && self.error.is_none()
    }

    /// Get the validation error, if any.
    #[must_use]
    pub fn error(&self) -> Option<&WarmError> {
        self.error.as_ref()
    }
}
