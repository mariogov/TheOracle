//! Feed-forward network (FFN) implementation for BERT encoder.
//!
//! Provides GPU-accelerated FFN with GELU activation.

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{BertConfig, FfnWeights};
use candle_core::Tensor;

/// Run FFN forward pass: intermediate -> GELU -> output
pub fn ffn_forward(
    hidden_states: &Tensor,
    ffn: &FfnWeights,
    config: &BertConfig,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len, hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("GraphModel layer {} FFN get dims failed: {}", layer_idx, e),
            })?;

    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel layer {} FFN flatten failed: {}", layer_idx, e),
        })?;

    // Intermediate projection
    let intermediate = hidden_flat
        .matmul(
            &ffn.intermediate_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "GraphModel layer {} FFN intermediate transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "GraphModel layer {} FFN intermediate matmul failed: {}",
                layer_idx, e
            ),
        })?;

    let intermediate = intermediate
        .broadcast_add(&ffn.intermediate_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "GraphModel layer {} FFN intermediate bias failed: {}",
                layer_idx, e
            ),
        })?;

    // GELU activation
    let intermediate = intermediate.gelu().map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel layer {} GELU failed: {}", layer_idx, e),
    })?;

    // Output projection
    let output = intermediate
        .matmul(
            &ffn.output_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "GraphModel layer {} FFN output transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "GraphModel layer {} FFN output matmul failed: {}",
                layer_idx, e
            ),
        })?;

    let output = output
        .broadcast_add(&ffn.output_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "GraphModel layer {} FFN output bias failed: {}",
                layer_idx, e
            ),
        })?;

    output
        .reshape((batch_size, seq_len, config.hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel layer {} FFN reshape failed: {}", layer_idx, e),
        })
}
