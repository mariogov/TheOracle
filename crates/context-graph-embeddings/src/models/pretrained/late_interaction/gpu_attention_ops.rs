//! Attention helper operations for ColBERT transformer.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

/// Dimensions for Q/K/V projection operations.
pub(crate) struct ProjectionDims {
    pub batch_size: usize,
    pub seq_len: usize,
    pub hidden_size: usize,
}

/// Compute Q/K/V projection.
pub(crate) fn compute_qkv_projection(
    hidden_flat: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    dims: &ProjectionDims,
    layer_idx: usize,
    name: &str,
) -> EmbeddingResult<Tensor> {
    let proj = hidden_flat
        .matmul(&weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} {} transpose failed: {}",
                layer_idx, name, e
            ),
        })?)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} {} matmul failed: {}",
                layer_idx, name, e
            ),
        })?
        .reshape((dims.batch_size, dims.seq_len, dims.hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} {} reshape failed: {}",
                layer_idx, name, e
            ),
        })?;

    proj.broadcast_add(bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} {} bias failed: {}",
                layer_idx, name, e
            ),
        })
}

/// Reshape for multi-head attention.
pub(crate) fn reshape_for_attention(
    tensor: &Tensor,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    layer_idx: usize,
    name: &str,
) -> EmbeddingResult<Tensor> {
    tensor
        .reshape((batch_size, seq_len, num_heads, head_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} {} head reshape failed: {}",
                layer_idx, name, e
            ),
        })?
        .transpose(1, 2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} {} transpose 1,2 failed: {}",
                layer_idx, name, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} {} contiguous failed: {}",
                layer_idx, name, e
            ),
        })
}

/// Compute attention scores and context.
pub(crate) fn compute_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attention_mask: &Tensor,
    head_dim: usize,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    let key_t = key
        .transpose(2, 3)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} K transpose 2,3 failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} K^T contiguous failed: {}",
                layer_idx, e
            ),
        })?;

    let scores = query.matmul(&key_t).map_err(|e| EmbeddingError::GpuError {
        message: format!(
            "LateInteractionModel layer {} QK matmul failed: {}",
            layer_idx, e
        ),
    })?;

    let scale = (head_dim as f64).sqrt();
    let scores = (scores / scale).map_err(|e| EmbeddingError::GpuError {
        message: format!(
            "LateInteractionModel layer {} attention scale failed: {}",
            layer_idx, e
        ),
    })?;

    let scores = scores
        .broadcast_add(attention_mask)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} attention mask add failed: {}",
                layer_idx, e
            ),
        })?;

    let attention_probs =
        candle_nn::ops::softmax(&scores, candle_core::D::Minus1).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!(
                    "LateInteractionModel layer {} softmax failed: {}",
                    layer_idx, e
                ),
            }
        })?;

    attention_probs
        .matmul(value)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} context matmul failed: {}",
                layer_idx, e
            ),
        })
}
