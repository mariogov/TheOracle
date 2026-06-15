//! Text construction tests for ModelInput.

use crate::error::EmbeddingError;
use crate::types::input::ModelInput;

// ============================================================
// TEXT CONSTRUCTION TESTS (5 tests)
// ============================================================

#[test]
fn test_text_with_valid_content_succeeds() {
    let input = ModelInput::text("Hello, world!");
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_text());
    let (content, instruction) = input.as_text().unwrap();
    assert_eq!(content, "Hello, world!");
    assert!(instruction.is_none());
}

#[test]
fn test_text_with_empty_content_returns_invalid_input_error() {
    println!("BEFORE: Attempting to create ModelInput::text(\"\")");
    let result = ModelInput::text("");
    println!("AFTER: Result = {:?}", result);

    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}

#[test]
fn test_text_with_instruction_succeeds() {
    let input = ModelInput::text_with_instruction("What is Rust?", "query:");
    assert!(input.is_ok());
    let input = input.unwrap();
    let (content, instruction) = input.as_text().unwrap();
    assert_eq!(content, "What is Rust?");
    assert_eq!(instruction, Some("query:"));
}

#[test]
fn test_text_with_instruction_stores_instruction_correctly() {
    let input = ModelInput::text_with_instruction("Document content", "passage:").unwrap();
    let (_, instruction) = input.as_text().unwrap();
    assert_eq!(instruction, Some("passage:"));

    // Empty instruction is allowed
    let input2 = ModelInput::text_with_instruction("Content", "").unwrap();
    let (_, instruction2) = input2.as_text().unwrap();
    assert_eq!(instruction2, Some(""));
}

#[test]
fn test_text_with_instruction_empty_content_returns_invalid_input() {
    let result = ModelInput::text_with_instruction("", "query:");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}
