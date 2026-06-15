//! Encoder layer and FFN forward pass for the sparse model.
//!
//! This module implements the BERT encoder layer components including
//! the feed-forward network (FFN) and layer normalization.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{BertConfig, EncoderLayerWeights, FfnWeights};

use super::attention::self_attention_forward;

/// Apply LayerNorm.
pub(crate) fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> EmbeddingResult<Tensor> {
    let mean = x
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm mean failed: {}", e),
        })?;

    let x_centered = x
        .broadcast_sub(&mean)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm center failed: {}", e),
        })?;

    let var = x_centered
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm sqr failed: {}", e),
        })?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm var mean failed: {}", e),
        })?;

    let std = (var + eps)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm var add eps failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm sqrt failed: {}", e),
        })?;

    let normalized = x_centered
        .broadcast_div(&std)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm div failed: {}", e),
        })?;

    let scaled = normalized
        .broadcast_mul(weight)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm scale failed: {}", e),
        })?;

    scaled
        .broadcast_add(bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel LayerNorm bias failed: {}", e),
        })
}

/// Run single encoder layer forward pass.
pub(crate) fn encoder_layer_forward(
    hidden_states: &Tensor,
    layer: &EncoderLayerWeights,
    attention_mask: &Tensor,
    config: &BertConfig,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    // Self-attention
    let attention_output = self_attention_forward(
        hidden_states,
        &layer.attention,
        attention_mask,
        config,
        layer_idx,
    )?;

    // Add & Norm (attention)
    let attention_output =
        (hidden_states + &attention_output).map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} attention residual failed: {}",
                layer_idx, e
            ),
        })?;

    let attention_output = layer_norm(
        &attention_output,
        &layer.attention.layer_norm_weight,
        &layer.attention.layer_norm_bias,
        config.layer_norm_eps,
    )?;

    // FFN
    let ffn_output = ffn_forward(&attention_output, &layer.ffn, config, layer_idx)?;

    // Add & Norm (FFN)
    let output = (&attention_output + &ffn_output).map_err(|e| EmbeddingError::GpuError {
        message: format!("SparseModel layer {} FFN residual failed: {}", layer_idx, e),
    })?;

    layer_norm(
        &output,
        &layer.ffn.layer_norm_weight,
        &layer.ffn.layer_norm_bias,
        config.layer_norm_eps,
    )
}

/// Run FFN forward pass.
pub(crate) fn ffn_forward(
    hidden_states: &Tensor,
    ffn: &FfnWeights,
    config: &BertConfig,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len, hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("SparseModel layer {} FFN get dims failed: {}", layer_idx, e),
            })?;

    let intermediate_size = config.intermediate_size;

    // Flatten for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} FFN flatten failed: {}", layer_idx, e),
        })?;

    let intermediate = hidden_flat
        .matmul(
            &ffn.intermediate_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "SparseModel layer {} FFN intermediate transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} FFN intermediate matmul failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} FFN intermediate reshape failed: {}",
                layer_idx, e
            ),
        })?;

    let intermediate = intermediate
        .broadcast_add(&ffn.intermediate_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} FFN intermediate bias failed: {}",
                layer_idx, e
            ),
        })?;

    let intermediate = intermediate.gelu().map_err(|e| EmbeddingError::GpuError {
        message: format!("SparseModel layer {} GELU failed: {}", layer_idx, e),
    })?;

    // Flatten again for output projection
    let intermediate_flat = intermediate
        .reshape((batch_size * seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} FFN intermediate flatten failed: {}",
                layer_idx, e
            ),
        })?;

    let output = intermediate_flat
        .matmul(
            &ffn.output_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "SparseModel layer {} FFN output transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} FFN output matmul failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} FFN output reshape failed: {}",
                layer_idx, e
            ),
        })?;

    output
        .broadcast_add(&ffn.output_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} FFN output bias failed: {}",
                layer_idx, e
            ),
        })
}
