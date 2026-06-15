//! Tests for embedding error types - Part 1: Basic tests.

use super::*;
use crate::types::{InputType, ModelId};
use std::error::Error;

// ============================================================
// MODEL ERROR CREATION TESTS (3 tests)
// ============================================================

#[test]
fn test_model_not_found_error_creation() {
    let err = EmbeddingError::ModelNotFound {
        model_id: ModelId::Semantic,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Semantic"));
    assert!(msg.contains("not found"));
}

#[test]
fn test_model_load_error_preserves_source() {
    let source = std::io::Error::new(std::io::ErrorKind::NotFound, "weights missing");
    let err = EmbeddingError::ModelLoadError {
        model_id: ModelId::Code,
        source: Box::new(source),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Code"));
    assert!(msg.contains("weights missing"));
    // Verify source chain
    assert!(err.source().is_some());
}

#[test]
fn test_not_initialized_error_creation() {
    let err = EmbeddingError::NotInitialized {
        model_id: ModelId::Graph,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Graph"));
    assert!(msg.contains("not initialized"));
}

// ============================================================
// VALIDATION ERROR CREATION TESTS (4 tests)
// ============================================================

#[test]
fn test_invalid_dimension_shows_both_values() {
    let err = EmbeddingError::InvalidDimension {
        expected: 1536,
        actual: 768,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("1536"));
    assert!(msg.contains("768"));
}

#[test]
fn test_invalid_value_shows_index_and_value() {
    let err = EmbeddingError::InvalidValue {
        index: 42,
        value: f32::NAN,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("42"));
    assert!(msg.contains("NaN"));
}

#[test]
fn test_empty_input_error_message() {
    let err = EmbeddingError::EmptyInput;
    let msg = format!("{}", err);
    assert!(msg.contains("Empty"));
}

#[test]
fn test_input_too_long_shows_limits() {
    let err = EmbeddingError::InputTooLong {
        actual: 600,
        max: 512,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("600"));
    assert!(msg.contains("512"));
}

// ============================================================
// PROCESSING ERROR CREATION TESTS (2 tests)
// ============================================================

#[test]
fn test_batch_error_shows_message() {
    let err = EmbeddingError::BatchError {
        message: "queue overflow".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Batch"));
    assert!(msg.contains("queue overflow"));
}

// NOTE: test_fusion_error_shows_message removed (TASK-F006)

#[test]
fn test_tokenization_error_shows_message() {
    let err = EmbeddingError::TokenizationError {
        model_id: ModelId::Semantic,
        message: "unknown token".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Tokenization"));
    assert!(msg.contains("unknown token"));
}

// ============================================================
// INFRASTRUCTURE ERROR CREATION TESTS (4 tests)
// ============================================================

#[test]
fn test_gpu_error_shows_message() {
    let err = EmbeddingError::GpuError {
        message: "CUDA OOM".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("GPU"));
    assert!(msg.contains("CUDA OOM"));
}

#[test]
fn test_cache_error_shows_message() {
    let err = EmbeddingError::CacheError {
        message: "eviction failed".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Cache"));
    assert!(msg.contains("eviction failed"));
}

#[test]
fn test_io_error_wraps_std_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let err = EmbeddingError::IoError(io_err);
    let msg = format!("{}", err);
    assert!(msg.contains("access denied"));
}

#[test]
fn test_timeout_error_shows_duration() {
    let err = EmbeddingError::Timeout { timeout_ms: 5000 };
    let msg = format!("{}", err);
    assert!(msg.contains("timeout"));
    assert!(msg.contains("5000"));
}

// ============================================================
// CONFIGURATION ERROR CREATION TESTS (2 tests)
// ============================================================

#[test]
fn test_unsupported_modality_shows_both() {
    let err = EmbeddingError::UnsupportedModality {
        model_id: ModelId::Semantic,
        input_type: InputType::Image,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Semantic"));
    assert!(msg.contains("Image"));
}

#[test]
fn test_config_error_shows_message() {
    let err = EmbeddingError::ConfigError {
        message: "missing required field".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Configuration"));
    assert!(msg.contains("missing required field"));
}

// ============================================================
// SERIALIZATION ERROR CREATION TEST (1 test)
// ============================================================

#[test]
fn test_serialization_error_shows_message() {
    let err = EmbeddingError::SerializationError {
        message: "invalid JSON".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Serialization"));
    assert!(msg.contains("invalid JSON"));
}

// ============================================================
// SEND + SYNC TESTS (2 tests)
// ============================================================

#[test]
fn test_embedding_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<EmbeddingError>();
}

#[test]
fn test_embedding_error_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<EmbeddingError>();
}

// ============================================================
// FROM<IO::ERROR> TEST (1 test)
// ============================================================

#[test]
fn test_io_error_conversion_via_question_mark() {
    fn fallible_io() -> EmbeddingResult<()> {
        let _ = std::fs::read("/nonexistent/path/that/does/not/exist/12345")?;
        Ok(())
    }
    let result = fallible_io();
    assert!(matches!(result, Err(EmbeddingError::IoError(_))));
}

// ============================================================
// EMBEDDING RESULT ALIAS TEST (1 test)
// ============================================================

#[test]
fn test_embedding_result_alias_works() {
    fn returns_ok() -> EmbeddingResult<i32> {
        Ok(42)
    }
    fn returns_err() -> EmbeddingResult<i32> {
        Err(EmbeddingError::EmptyInput)
    }
    assert_eq!(returns_ok().unwrap(), 42);
    assert!(returns_err().is_err());
}

// ============================================================
// DEBUG FORMATTING TEST (1 test)
// ============================================================

#[test]
fn test_debug_formatting_includes_variant_name() {
    let err = EmbeddingError::Timeout { timeout_ms: 5000 };
    let debug = format!("{:?}", err);
    assert!(debug.contains("Timeout"));
    assert!(debug.contains("5000"));
}
