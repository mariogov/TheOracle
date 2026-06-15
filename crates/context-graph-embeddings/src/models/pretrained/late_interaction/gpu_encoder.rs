//! GPU encoder layers for ColBERT transformer.

use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{BertConfig, BertWeights, EncoderLayerWeights};

use super::gpu_attention::self_attention_forward;
use super::gpu_utils::{ffn_forward, layer_norm};

/// Run all encoder layers with attention mask.
pub(crate) fn run_encoder_layers(
    mut hidden_states: Tensor,
    attention_mask_tensor: &Tensor,
    weights: &BertWeights,
    config: &BertConfig,
) -> EmbeddingResult<Tensor> {
    // Create attention mask for broadcasting: [batch, 1, 1, seq_len]
    let extended_attention_mask = attention_mask_tensor
        .unsqueeze(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel attention mask unsqueeze 1 failed: {}",
                e
            ),
        })?
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel attention mask unsqueeze 2 failed: {}",
                e
            ),
        })?;

    // Convert mask: 1.0 -> 0.0, 0.0 -> -10000.0
    let extended_attention_mask =
        ((extended_attention_mask * (-1.0)).map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel attention mask mul failed: {}", e),
        })? + 1.0)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("LateInteractionModel attention mask add failed: {}", e),
            })?
            * (-10000.0f64);

    let extended_attention_mask =
        extended_attention_mask.map_err(|e| EmbeddingError::GpuError {
            message: format!("LateInteractionModel attention mask scale failed: {}", e),
        })?;

    for (layer_idx, layer) in weights.encoder_layers.iter().enumerate() {
        hidden_states = encoder_layer_forward(
            &hidden_states,
            layer,
            &extended_attention_mask,
            config,
            layer_idx,
        )?;
    }

    Ok(hidden_states)
}

/// Run single encoder layer forward pass.
fn encoder_layer_forward(
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
        config.hidden_size,
        config.num_attention_heads,
        layer_idx,
    )?;

    // Add & Norm (attention)
    let attention_output =
        (hidden_states + &attention_output).map_err(|e| EmbeddingError::GpuError {
            message: format!(
                "LateInteractionModel layer {} attention residual failed: {}",
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
        message: format!(
            "LateInteractionModel layer {} FFN residual failed: {}",
            layer_idx, e
        ),
    })?;

    layer_norm(
        &output,
        &layer.ffn.layer_norm_weight,
        &layer.ffn.layer_norm_bias,
        config.layer_norm_eps,
    )
}
