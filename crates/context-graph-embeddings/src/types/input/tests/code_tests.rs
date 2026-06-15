//! Code construction tests for ModelInput.

use crate::error::EmbeddingError;
use crate::types::input::ModelInput;

// ============================================================
// CODE CONSTRUCTION TESTS (4 tests)
// ============================================================

#[test]
fn test_code_with_valid_content_and_language_succeeds() {
    let input = ModelInput::code("fn main() {}", "rust");
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_code());
    let (content, language) = input.as_code().unwrap();
    assert_eq!(content, "fn main() {}");
    assert_eq!(language, "rust");
}

#[test]
fn test_code_with_empty_content_returns_empty_input() {
    let result = ModelInput::code("", "rust");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error for empty content"
    );
}

#[test]
fn test_code_with_empty_language_returns_config_error() {
    let result = ModelInput::code("fn main() {}", "");
    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("language") && message.contains("empty"));
        }
        _ => panic!("Expected ConfigError for empty language"),
    }
}

#[test]
fn test_code_stores_language_as_provided() {
    // Language is stored as-is (no normalization)
    let input = ModelInput::code("code", "Rust").unwrap();
    let (_, language) = input.as_code().unwrap();
    assert_eq!(language, "Rust");

    let input2 = ModelInput::code("code", "PYTHON").unwrap();
    let (_, language2) = input2.as_code().unwrap();
    assert_eq!(language2, "PYTHON");
}
