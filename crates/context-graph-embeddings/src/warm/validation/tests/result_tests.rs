//! Tests for ValidationResult.

use crate::warm::error::WarmError;
use crate::warm::validation::ValidationResult;

#[test]
fn test_validation_result_success() {
    let result = ValidationResult::success("model1".to_string(), 42);

    assert_eq!(result.model_id, "model1");
    assert!(result.dimension_valid);
    assert!(result.weights_valid);
    assert!(result.inference_valid);
    assert_eq!(result.inference_time_ms, 42);
    assert!(result.is_valid());
    assert!(result.error().is_none());
}

#[test]
fn test_validation_result_failure() {
    let error = WarmError::ModelDimensionMismatch {
        model_id: "model1".to_string(),
        expected: 100,
        actual: 50,
    };

    let result = ValidationResult::failure("model1".to_string(), false, true, false, 100, error);

    assert!(!result.dimension_valid);
    assert!(result.weights_valid);
    assert!(!result.inference_valid);
    assert!(!result.is_valid());
    assert!(result.error().is_some());
}
