//! GPU utility functions for ColBERT late-interaction model.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{BertConfig, FfnWeights};

/// Apply LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
pub(crate) fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> EmbeddingResult<Tensor> {
    let mean = x
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm mean failed: {}", e),
        })?;

    let x_centered = x
        .broadcast_sub(&mean)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm center failed: {}", e),
        })?;

    let var = x_centered
        .sqr()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm sqr failed: {}", e),
        })?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm var mean failed: {}", e),
        })?;

    let std = (var + eps)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm var add eps failed: {}", e),
        })?
        .sqrt()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm sqrt failed: {}", e),
        })?;

    let normalized = x_centered
        .broadcast_div(&std)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm div failed: {}", e),
        })?;

    let scaled = normalized
        .broadcast_mul(weight)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm scale failed: {}", e),
        })?;

    scaled
        .broadcast_add(bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel LayerNorm bias failed: {}", e),
        })
}

/// Run FFN forward pass: intermediate -> GELU -> output
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
                message: format!(
                    "LateInteractionModel layer {} FFN get dims failed: {}",
                    layer_idx, e
                ),
            })?;

    let intermediate_size = config.intermediate_size;

    // Flatten for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN flatten failed: {}",
                layer_idx, e
            ),
        })?;

    // Intermediate projection
    let intermediate = hidden_flat
        .matmul(
            &ffn.intermediate_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "LateInteractionModel layer {} FFN intermediate transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN intermediate matmul failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN intermediate reshape failed: {}",
                layer_idx, e
            ),
        })?;

    let intermediate = intermediate
        .broadcast_add(&ffn.intermediate_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN intermediate bias failed: {}",
                layer_idx, e
            ),
        })?;

    // GELU activation
    let intermediate = intermediate.gelu().map_err(|e| EmbeddingError::GpuError {
        message: format!(
            "LateInteractionModel layer {} GELU failed: {}",
            layer_idx, e
        ),
    })?;

    // Flatten for output projection
    let intermediate_flat = intermediate
        .reshape((batch_size * seq_len, intermediate_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN intermediate flatten failed: {}",
                layer_idx, e
            ),
        })?;

    // Output projection
    let output = intermediate_flat
        .matmul(
            &ffn.output_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "LateInteractionModel layer {} FFN output transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN output matmul failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN output reshape failed: {}",
                layer_idx, e
            ),
        })?;

    output
        .broadcast_add(&ffn.output_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} FFN output bias failed: {}",
                layer_idx, e
            ),
        })
}
