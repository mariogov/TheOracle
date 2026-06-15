//! Tests for TestInferenceConfig and TestInput.

use crate::warm::validation::{TestInferenceConfig, TestInput};

// === TestInferenceConfig Tests ===

#[test]
fn test_config_for_embedding_model() {
    let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);

    assert_eq!(config.model_id, "E1_Semantic");
    assert_eq!(config.expected_dimension, 1024);
    assert!(matches!(config.test_input, TestInput::Text(_)));
    assert!(config.reference_output.is_none());
    assert_eq!(config.max_inference_ms, 1000);
}

#[test]
fn test_config_with_reference() {
    let reference = vec![0.1, 0.2, 0.3];
    let config = TestInferenceConfig::with_reference(
        "TestModel",
        3,
        TestInput::Text("test".to_string()),
        reference.clone(),
        500,
    );

    assert_eq!(config.model_id, "TestModel");
    assert_eq!(config.expected_dimension, 3);
    assert_eq!(config.reference_output.as_ref().unwrap(), &reference);
    assert_eq!(config.max_inference_ms, 500);
}

// === TestInput Tests ===

#[test]
fn test_input_description() {
    assert_eq!(TestInput::Text("hello".to_string()).description(), "text");
    assert_eq!(TestInput::Tokens(vec![1, 2, 3]).description(), "tokens");
    assert_eq!(
        TestInput::Embeddings(vec![0.1, 0.2]).description(),
        "embeddings"
    );
}

#[test]
fn test_input_len() {
    assert_eq!(TestInput::Text("hello".to_string()).len(), 5);
    assert_eq!(TestInput::Tokens(vec![1, 2, 3]).len(), 3);
    assert_eq!(TestInput::Embeddings(vec![0.1, 0.2]).len(), 2);
}

#[test]
fn test_input_is_empty() {
    assert!(!TestInput::Text("hello".to_string()).is_empty());
    assert!(TestInput::Text(String::new()).is_empty());
    assert!(TestInput::Tokens(vec![]).is_empty());
    assert!(!TestInput::Embeddings(vec![0.1]).is_empty());
}
