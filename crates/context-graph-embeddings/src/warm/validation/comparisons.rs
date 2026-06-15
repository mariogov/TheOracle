//! Output comparison logic for validation.
//!
//! Internal implementation for output comparison. Public methods are
//! exposed through `WarmValidator` in `validator.rs`.

use crate::warm::error::{WarmError, WarmResult};

/// Internal implementation for output comparison.
///
/// Used by `WarmValidator::compare_output` and `WarmValidator::compare_output_for_model`.
pub(super) fn compare_output_impl(
    model_id: &str,
    actual: &[f32],
    reference: &[f32],
    tolerance: f32,
) -> WarmResult<()> {
    if actual.len() != reference.len() {
        return Err(WarmError::ModelValidationFailed {
            model_id: model_id.to_string(),
            reason: format!(
                "Output length mismatch: expected {}, got {}",
                reference.len(),
                actual.len()
            ),
            expected_output: Some(format!("length {}", reference.len())),
            actual_output: Some(format!("length {}", actual.len())),
        });
    }

    for (idx, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
        let diff = (a - r).abs();
        if diff > tolerance {
            return Err(WarmError::ModelValidationFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "Output mismatch at index {idx}: expected {r}, got {a} \
                     (diff: {diff}, tolerance: {tolerance})"
                ),
                expected_output: Some(format!("{r}")),
                actual_output: Some(format!("{a}")),
            });
        }
    }

    Ok(())
}
