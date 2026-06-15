//! Mean pooling and normalization for semantic model.
//!
//! Applies mean pooling over sequence dimension and L2 normalization.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::normalize_gpu;

/// Apply mean pooling and L2 normalization.
pub fn pool_and_normalize(
    hidden_states: &Tensor,
    attention_mask_tensor: &Tensor,
    seq_len: usize,
    hidden_size: usize,
) -> EmbeddingResult<Vec<f32>> {
    let mask_expanded = attention_mask_tensor
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel mask expand failed: {}", e),
        })?
        .broadcast_as((1, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel mask broadcast failed: {}", e),
        })?;

    let masked_hidden = (hidden_states * mask_expanded).map_err(|e| EmbeddingError::GpuError {
        message: format!("SemanticModel masked multiply failed: {}", e),
    })?;

    let sum_hidden = masked_hidden.sum(1).map_err(|e| EmbeddingError::GpuError {
        message: format!("SemanticModel sum hidden failed: {}", e),
    })?;

    let mask_sum = attention_mask_tensor
        .sum(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel mask sum failed: {}", e),
        })?
        .unsqueeze(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel mask sum unsqueeze failed: {}", e),
        })?
        .broadcast_as(sum_hidden.shape())
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel mask sum broadcast failed: {}", e),
        })?;

    let pooled = (sum_hidden
        / (mask_sum + 1e-9f64).map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel mask sum add eps failed: {}", e),
        })?)
    .map_err(|e| EmbeddingError::GpuError {
        message: format!("SemanticModel mean pooling div failed: {}", e),
    })?;

    // L2 normalize
    let normalized = normalize_gpu(&pooled).map_err(|e| EmbeddingError::GpuError {
        message: format!("SemanticModel L2 normalize failed: {}", e),
    })?;

    // Convert to Vec<f32>
    normalized
        .flatten_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel flatten output failed: {}", e),
        })?
        .to_vec1()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel to_vec1 failed: {}", e),
        })
}
