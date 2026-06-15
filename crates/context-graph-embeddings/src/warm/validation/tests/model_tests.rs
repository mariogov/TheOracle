//! Tests for WarmValidator::validate_model integration.

use crate::warm::error::WarmError;
use crate::warm::handle::ModelHandle;
use crate::warm::validation::{TestInferenceConfig, TestInput, WarmValidator};

#[test]
fn test_validate_model_success() {
    let v = WarmValidator::new();
    let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 4);
    let handle = ModelHandle::new(0x1000, 4 * 4, 0, 0xabcd);
    let output = vec![0.1, 0.2, 0.3, 0.4];

    let result = v.validate_model(&config, &handle, &output);

    assert!(result.is_valid());
    assert!(result.dimension_valid);
    assert!(result.weights_valid);
    assert!(result.inference_valid);
    assert!(result.error.is_none());
}

#[test]
fn test_validate_model_dimension_failure() {
    let v = WarmValidator::new();
    let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);
    let handle = ModelHandle::new(0x1000, 512 * 4, 0, 0xabcd);
    let output = vec![0.1; 512];

    let result = v.validate_model(&config, &handle, &output);

    assert!(!result.is_valid());
    assert!(!result.dimension_valid);
    assert!(matches!(
        result.error,
        Some(WarmError::ModelDimensionMismatch { .. })
    ));
}

#[test]
fn test_validate_model_nan_failure() {
    let v = WarmValidator::new();
    let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 4);
    let handle = ModelHandle::new(0x1000, 4 * 4, 0, 0xabcd);
    let output = vec![0.1, f32::NAN, 0.3, 0.4];

    let result = v.validate_model(&config, &handle, &output);

    assert!(!result.is_valid());
    assert!(result.dimension_valid);
    assert!(!result.weights_valid);
    assert!(matches!(
        result.error,
        Some(WarmError::ModelValidationFailed { .. })
    ));
}

#[test]
fn test_validate_model_reference_mismatch() {
    let v = WarmValidator::new();
    let reference = vec![0.1, 0.2, 0.3, 0.4];
    let config = TestInferenceConfig::with_reference(
        "TestModel",
        4,
        TestInput::Text("test".to_string()),
        reference,
        1000,
    );
    let handle = ModelHandle::new(0x1000, 4 * 4, 0, 0xabcd);
    let output = vec![0.1, 0.5, 0.3, 0.4]; // 0.5 != 0.2

    let result = v.validate_model(&config, &handle, &output);

    assert!(!result.is_valid());
    assert!(result.dimension_valid);
    assert!(result.weights_valid);
    assert!(!result.inference_valid);
}

#[test]
fn test_validate_model_with_inf() {
    let v = WarmValidator::new();
    let config = TestInferenceConfig::for_embedding_model("TestModel", 3);
    let handle = ModelHandle::new(0x2000, 3 * 4, 0, 0x1234);
    let output = vec![0.1, f32::INFINITY, 0.3];

    let result = v.validate_model(&config, &handle, &output);

    assert!(!result.is_valid());
    assert!(!result.weights_valid);
}

#[test]
fn test_validation_result_aggregation() {
    // Test that ValidationResult properly aggregates multiple validation stages
    let v = WarmValidator::new();
    let config = TestInferenceConfig::for_embedding_model("Aggregation_Test", 3);
    let handle = ModelHandle::new(0x1000, 12, 0, 0xfeed);

    // All valid
    let output1 = vec![0.1, 0.2, 0.3];
    let r1 = v.validate_model(&config, &handle, &output1);
    assert!(r1.is_valid());
    assert!(r1.dimension_valid && r1.weights_valid && r1.inference_valid);

    // Wrong dimension - should fail early
    let output2 = vec![0.1, 0.2];
    let r2 = v.validate_model(&config, &handle, &output2);
    assert!(!r2.is_valid());
    assert!(!r2.dimension_valid);
    // Note: weights_valid may be true since we didn't check (early fail)

    // Valid dimension, invalid weights
    let output3 = vec![0.1, f32::NAN, 0.3];
    let r3 = v.validate_model(&config, &handle, &output3);
    assert!(!r3.is_valid());
    assert!(r3.dimension_valid);
    assert!(!r3.weights_valid);
}
