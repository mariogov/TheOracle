//! Private encoder methods for QuantizationRouter.
//!
//! Contains the encoder-specific quantization and dequantization logic
//! for Binary, Float8, and PQ8 methods.

use super::QuantizationRouter;
use crate::error::EmbeddingError;
use crate::quantization::binary::BinaryQuantizationError;
use crate::quantization::float8::Float8QuantizationError;
use crate::quantization::pq8::PQ8QuantizationError;
use crate::quantization::types::QuantizedEmbedding;
use crate::types::ModelId;
use tracing::error;

impl QuantizationRouter {
    // =========================================================================
    // Binary encoder methods
    // =========================================================================

    /// Quantize using binary encoder.
    pub(super) fn quantize_binary(
        &self,
        model_id: ModelId,
        embedding: &[f32],
    ) -> Result<QuantizedEmbedding, EmbeddingError> {
        self.binary_encoder
            .quantize(embedding, Some(0.0))
            .map_err(|e| {
                error!(
                    target: "quantization::router",
                    model_id = ?model_id,
                    error = %e,
                    "Binary quantization failed"
                );
                Self::binary_error_to_embedding_error(model_id, e, true)
            })
    }

    /// Dequantize using binary decoder.
    pub(super) fn dequantize_binary(
        &self,
        model_id: ModelId,
        quantized: &QuantizedEmbedding,
    ) -> Result<Vec<f32>, EmbeddingError> {
        self.binary_encoder.dequantize(quantized).map_err(|e| {
            error!(
                target: "quantization::router",
                model_id = ?model_id,
                error = %e,
                "Binary dequantization failed"
            );
            Self::binary_error_to_embedding_error(model_id, e, false)
        })
    }

    /// Convert BinaryQuantizationError to EmbeddingError.
    fn binary_error_to_embedding_error(
        model_id: ModelId,
        error: BinaryQuantizationError,
        is_quantization: bool,
    ) -> EmbeddingError {
        let reason = error.to_string();
        if is_quantization {
            EmbeddingError::QuantizationFailed { model_id, reason }
        } else {
            EmbeddingError::DequantizationFailed { model_id, reason }
        }
    }

    // =========================================================================
    // Float8 encoder methods
    // =========================================================================

    /// Quantize using Float8 E4M3 encoder.
    pub(super) fn quantize_float8(
        &self,
        model_id: ModelId,
        embedding: &[f32],
    ) -> Result<QuantizedEmbedding, EmbeddingError> {
        self.float8_encoder.quantize(embedding).map_err(|e| {
            error!(
                target: "quantization::router",
                model_id = ?model_id,
                error = %e,
                "Float8E4M3 quantization failed"
            );
            Self::float8_error_to_embedding_error(model_id, e, true)
        })
    }

    /// Dequantize using Float8 E4M3 decoder.
    pub(super) fn dequantize_float8(
        &self,
        model_id: ModelId,
        quantized: &QuantizedEmbedding,
    ) -> Result<Vec<f32>, EmbeddingError> {
        self.float8_encoder.dequantize(quantized).map_err(|e| {
            error!(
                target: "quantization::router",
                model_id = ?model_id,
                error = %e,
                "Float8E4M3 dequantization failed"
            );
            Self::float8_error_to_embedding_error(model_id, e, false)
        })
    }

    /// Convert Float8QuantizationError to EmbeddingError.
    fn float8_error_to_embedding_error(
        model_id: ModelId,
        error: Float8QuantizationError,
        is_quantization: bool,
    ) -> EmbeddingError {
        let reason = error.to_string();
        if is_quantization {
            EmbeddingError::QuantizationFailed { model_id, reason }
        } else {
            EmbeddingError::DequantizationFailed { model_id, reason }
        }
    }

    // =========================================================================
    // PQ8 encoder methods
    // =========================================================================

    /// Quantize using PQ-8 encoder.
    pub(super) fn quantize_pq8(
        &self,
        model_id: ModelId,
        embedding: &[f32],
    ) -> Result<QuantizedEmbedding, EmbeddingError> {
        let dim = embedding.len();

        // Get or create encoder for this dimension
        let encoder = self.pq8_encoders.get(&dim).ok_or_else(|| {
            error!(
                target: "quantization::router",
                model_id = ?model_id,
                dim = dim,
                "No PQ8 encoder for dimension"
            );
            EmbeddingError::QuantizationFailed {
                model_id,
                reason: format!("No PQ8 encoder for dimension {}", dim),
            }
        })?;

        encoder.quantize(embedding).map_err(|e| {
            error!(
                target: "quantization::router",
                model_id = ?model_id,
                error = %e,
                "PQ8 quantization failed"
            );
            Self::pq8_error_to_embedding_error(model_id, e, true)
        })
    }

    /// Dequantize using PQ-8 decoder.
    pub(super) fn dequantize_pq8(
        &self,
        model_id: ModelId,
        quantized: &QuantizedEmbedding,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let dim = quantized.original_dim;

        // Get encoder for this dimension
        let encoder = self.pq8_encoders.get(&dim).ok_or_else(|| {
            error!(
                target: "quantization::router",
                model_id = ?model_id,
                dim = dim,
                "No PQ8 encoder for dimension"
            );
            EmbeddingError::DequantizationFailed {
                model_id,
                reason: format!("No PQ8 encoder for dimension {}", dim),
            }
        })?;

        encoder.dequantize(quantized).map_err(|e| {
            error!(
                target: "quantization::router",
                model_id = ?model_id,
                error = %e,
                "PQ8 dequantization failed"
            );
            Self::pq8_error_to_embedding_error(model_id, e, false)
        })
    }

    /// Convert PQ8QuantizationError to EmbeddingError.
    fn pq8_error_to_embedding_error(
        model_id: ModelId,
        error: PQ8QuantizationError,
        is_quantization: bool,
    ) -> EmbeddingError {
        let reason = error.to_string();
        if is_quantization {
            EmbeddingError::QuantizationFailed { model_id, reason }
        } else {
            EmbeddingError::DequantizationFailed { model_id, reason }
        }
    }
}
