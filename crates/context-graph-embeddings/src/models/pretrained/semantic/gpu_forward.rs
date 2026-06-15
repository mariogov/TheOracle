//! GPU-accelerated BERT forward pass for semantic embeddings.
//!
//! This module implements the complete forward pass pipeline:
//! 1. Tokenization
//! 2. Embedding lookup (word + position + token_type)
//! 3. Transformer encoder layers
//! 4. Mean pooling
//! 5. L2 normalization

use candle_core::Tensor;
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::BertWeights;
use crate::types::ModelId;

use super::constants::SEMANTIC_MAX_TOKENS;
use super::embeddings::compute_embeddings;
use super::encoder::encoder_layer_forward;
use super::pooling::pool_and_normalize;

/// Run GPU-accelerated BERT forward pass.
///
/// # GPU Pipeline
///
/// 1. Tokenize input text to token IDs
/// 2. Create GPU tensors for input_ids, attention_mask, token_type_ids
/// 3. Embedding lookup: word + position + token_type
/// 4. Apply LayerNorm to embeddings
/// 5. Run transformer encoder layers (self-attention + FFN)
/// 6. Mean pooling over sequence length
/// 7. L2 normalization
pub fn gpu_forward(
    text: &str,
    weights: &BertWeights,
    tokenizer: &Tokenizer,
) -> EmbeddingResult<Vec<f32>> {
    let device = weights.device();
    let config = &weights.config;

    // Tokenize input text
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| EmbeddingError::TokenizationError {
            model_id: ModelId::Semantic,
            message: format!("SemanticModel tokenization failed: {}", e),
        })?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<f32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as f32)
        .collect();

    // Truncate to max_position_embeddings if needed
    let max_len = config.max_position_embeddings.min(SEMANTIC_MAX_TOKENS);
    let seq_len = token_ids.len().min(max_len);
    let token_ids = &token_ids[..seq_len];
    let attention_mask = &attention_mask[..seq_len];

    // Create GPU tensors
    let input_ids = Tensor::from_slice(token_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("SemanticModel input_ids tensor failed: {}", e),
        }
    })?;

    let attention_mask_tensor =
        Tensor::from_slice(attention_mask, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("SemanticModel attention_mask tensor failed: {}", e),
            }
        })?;

    // Compute embeddings
    let embeddings = compute_embeddings(&input_ids, seq_len, weights, config, device)?;

    // Run encoder layers
    let hidden_states = run_encoder_layers(embeddings, &attention_mask_tensor, weights, config)?;

    // Pooling and normalization
    pool_and_normalize(
        &hidden_states,
        &attention_mask_tensor,
        seq_len,
        config.hidden_size,
    )
}

/// Run all encoder layers.
fn run_encoder_layers(
    embeddings: Tensor,
    attention_mask_tensor: &Tensor,
    weights: &BertWeights,
    config: &crate::gpu::BertConfig,
) -> EmbeddingResult<Tensor> {
    let mut hidden_states = embeddings;

    // Create attention mask for broadcasting: [batch, 1, 1, seq_len]
    let extended_attention_mask = attention_mask_tensor
        .unsqueeze(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel attention mask unsqueeze 1 failed: {}", e),
        })?
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel attention mask unsqueeze 2 failed: {}", e),
        })?;

    // Convert mask: 1.0 -> 0.0, 0.0 -> -10000.0
    let extended_attention_mask =
        ((extended_attention_mask * (-1.0)).map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel attention mask mul failed: {}", e),
        })? + 1.0)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("SemanticModel attention mask add failed: {}", e),
            })?
            * (-10000.0f64);

    let extended_attention_mask =
        extended_attention_mask.map_err(|e| EmbeddingError::GpuError {
            message: format!("SemanticModel attention mask scale failed: {}", e),
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
