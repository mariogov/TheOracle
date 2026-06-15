//! Self-attention forward pass for the sparse model.
//!
//! This module implements the multi-head self-attention mechanism used
//! in the BERT encoder layers of the SPLADE model.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{AttentionWeights, BertConfig};

/// Run self-attention forward pass.
pub(crate) fn self_attention_forward(
    hidden_states: &Tensor,
    attention: &AttentionWeights,
    attention_mask: &Tensor,
    config: &BertConfig,
    layer_idx: usize,
) -> EmbeddingResult<Tensor> {
    let (batch_size, seq_len, _hidden_size) =
        hidden_states
            .dims3()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("SparseModel layer {} get dims failed: {}", layer_idx, e),
            })?;

    let head_dim = config.hidden_size / config.num_attention_heads;
    let hidden_size = config.hidden_size;

    // Flatten to [batch*seq, hidden] for Candle matmul compatibility
    let hidden_flat = hidden_states
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} flatten hidden failed: {}",
                layer_idx, e
            ),
        })?;

    // Q, K, V projections with flatten/reshape pattern
    let query = hidden_flat
        .matmul(
            &attention
                .query_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("SparseModel layer {} Q transpose failed: {}", layer_idx, e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} Q matmul failed: {}", layer_idx, e),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} Q reshape failed: {}", layer_idx, e),
        })?;
    let query =
        query
            .broadcast_add(&attention.query_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("SparseModel layer {} Q bias failed: {}", layer_idx, e),
            })?;

    let key = hidden_flat
        .matmul(
            &attention
                .key_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("SparseModel layer {} K transpose failed: {}", layer_idx, e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} K matmul failed: {}", layer_idx, e),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} K reshape failed: {}", layer_idx, e),
        })?;
    let key = key
        .broadcast_add(&attention.key_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} K bias failed: {}", layer_idx, e),
        })?;

    let value = hidden_flat
        .matmul(
            &attention
                .value_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("SparseModel layer {} V transpose failed: {}", layer_idx, e),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} V matmul failed: {}", layer_idx, e),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} V reshape failed: {}", layer_idx, e),
        })?;
    let value =
        value
            .broadcast_add(&attention.value_bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("SparseModel layer {} V bias failed: {}", layer_idx, e),
            })?;

    // Reshape to [batch, heads, seq_len, head_dim] with contiguous after transpose
    let query = query
        .reshape((batch_size, seq_len, config.num_attention_heads, head_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} Q head reshape failed: {}",
                layer_idx, e
            ),
        })?
        .transpose(1, 2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} Q transpose 1,2 failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} Q contiguous failed: {}", layer_idx, e),
        })?;

    let key = key
        .reshape((batch_size, seq_len, config.num_attention_heads, head_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} K head reshape failed: {}",
                layer_idx, e
            ),
        })?
        .transpose(1, 2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} K transpose 1,2 failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} K contiguous failed: {}", layer_idx, e),
        })?;

    let value = value
        .reshape((batch_size, seq_len, config.num_attention_heads, head_dim))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} V head reshape failed: {}",
                layer_idx, e
            ),
        })?
        .transpose(1, 2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} V transpose 1,2 failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} V contiguous failed: {}", layer_idx, e),
        })?;

    // Attention scores with contiguous K^T
    let key_t = key
        .transpose(2, 3)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} K transpose 2,3 failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} K^T contiguous failed: {}",
                layer_idx, e
            ),
        })?;

    let scores = query.matmul(&key_t).map_err(|e| EmbeddingError::GpuError {
        message: format!("SparseModel layer {} QK matmul failed: {}", layer_idx, e),
    })?;

    let scale = (head_dim as f64).sqrt();
    let scores = (scores / scale).map_err(|e| EmbeddingError::GpuError {
        message: format!(
            "SparseModel layer {} attention scale failed: {}",
            layer_idx, e
        ),
    })?;

    // Use broadcast_add for attention mask
    let scores = scores
        .broadcast_add(attention_mask)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} attention mask add failed: {}",
                layer_idx, e
            ),
        })?;

    let attention_probs =
        candle_nn::ops::softmax(&scores, candle_core::D::Minus1).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("SparseModel layer {} softmax failed: {}", layer_idx, e),
            }
        })?;

    let context = attention_probs
        .matmul(&value)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} context matmul failed: {}",
                layer_idx, e
            ),
        })?;

    // Reshape context back with contiguous
    let context = context
        .transpose(1, 2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} context transpose failed: {}",
                layer_idx, e
            ),
        })?
        .contiguous()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} context contiguous failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} context reshape failed: {}",
                layer_idx, e
            ),
        })?;

    // Output projection with flatten/reshape
    let context_flat = context
        .reshape((batch_size * seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} context flatten failed: {}",
                layer_idx, e
            ),
        })?;

    let output = context_flat
        .matmul(
            &attention
                .output_weight
                .t()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!(
                        "SparseModel layer {} output transpose failed: {}",
                        layer_idx, e
                    ),
                })?,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} output matmul failed: {}",
                layer_idx, e
            ),
        })?
        .reshape((batch_size, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "SparseModel layer {} output reshape failed: {}",
                layer_idx, e
            ),
        })?;

    output
        .broadcast_add(&attention.output_bias)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel layer {} output bias failed: {}", layer_idx, e),
        })
}
