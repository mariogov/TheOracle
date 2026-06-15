//! Error handling tests for QuantizationRouter.

use crate::error::EmbeddingError;
use crate::quantization::router::QuantizationRouter;
use crate::quantization::types::{QuantizationMetadata, QuantizationMethod, QuantizedEmbedding};
use crate::types::ModelId;

// =========================================================================
// Invalid path tests
// =========================================================================

#[test]
fn test_sparse_rejects_dense() {
    let router = QuantizationRouter::new();

    // E6_Sparse should NOT use dense quantization path
    let embedding = vec![0.0f32; 30522]; // Sparse vocab size
    let result = router.quantize(ModelId::Sparse, &embedding);

    assert!(result.is_err());
    match result.unwrap_err() {
        EmbeddingError::InvalidModelInput { model_id, reason } => {
            assert_eq!(model_id, ModelId::Sparse);
            assert!(reason.contains("Sparse"));
        }
        e => panic!("Expected InvalidModelInput, got {:?}", e),
    }
}

#[test]
fn test_token_pruning_unsupported() {
    let router = QuantizationRouter::new();

    // E12_LateInteraction uses TokenPruning - out of scope
    let embedding = vec![0.5f32; 128];
    let result = router.quantize(ModelId::LateInteraction, &embedding);

    assert!(result.is_err());
    match result.unwrap_err() {
        EmbeddingError::UnsupportedOperation {
            model_id,
            operation,
        } => {
            assert_eq!(model_id, ModelId::LateInteraction);
            assert!(operation.contains("TokenPruning"));
        }
        e => panic!("Expected UnsupportedOperation, got {:?}", e),
    }
}

// =========================================================================
// Error handling tests
// =========================================================================

#[test]
fn test_binary_quantization_empty_input() {
    let router = QuantizationRouter::new();

    let result = router.quantize(ModelId::Hdc, &[]);

    assert!(result.is_err());
    match result.unwrap_err() {
        EmbeddingError::QuantizationFailed { model_id, reason } => {
            assert_eq!(model_id, ModelId::Hdc);
            assert!(reason.contains("Empty"));
        }
        e => panic!("Expected QuantizationFailed, got {:?}", e),
    }
}

#[test]
fn test_binary_quantization_nan_input() {
    let router = QuantizationRouter::new();

    let embedding = vec![1.0, f32::NAN, 0.5];
    let result = router.quantize(ModelId::Hdc, &embedding);

    assert!(result.is_err());
    match result.unwrap_err() {
        EmbeddingError::QuantizationFailed { model_id, reason } => {
            assert_eq!(model_id, ModelId::Hdc);
            assert!(reason.contains("NaN") || reason.contains("Invalid"));
        }
        e => panic!("Expected QuantizationFailed, got {:?}", e),
    }
}

#[test]
fn test_binary_dequantization_wrong_metadata() {
    let router = QuantizationRouter::new();

    // Create a QuantizedEmbedding with wrong metadata type
    let bad_quantized = QuantizedEmbedding {
        method: QuantizationMethod::Binary,
        original_dim: 8,
        data: vec![0xFF],
        metadata: QuantizationMetadata::Float8 {
            scale: 1.0,
            bias: 0.0,
        },
    };

    let result = router.dequantize(ModelId::Hdc, &bad_quantized);

    assert!(result.is_err());
    match result.unwrap_err() {
        EmbeddingError::DequantizationFailed { model_id, reason } => {
            assert_eq!(model_id, ModelId::Hdc);
            assert!(reason.contains("metadata") || reason.contains("Binary"));
        }
        e => panic!("Expected DequantizationFailed, got {:?}", e),
    }
}
