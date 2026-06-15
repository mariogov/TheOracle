//! GPU forward pass implementation for GraphModel.
//!
//! Implements the full BERT forward pass with embedding, encoder layers, and pooling.

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::{normalize_gpu, BertWeights};
use crate::types::ModelId;
use candle_core::Tensor;
use tokenizers::Tokenizer;

use super::constants::GRAPH_MAX_TOKENS;
use super::encoder::encoder_layer_forward;
use super::layer_norm::layer_norm;

/// Run GPU-accelerated BERT forward pass.
///
/// # GPU Pipeline
///
/// 1. Tokenize input text to token IDs
/// 2. Create GPU tensors for input_ids, attention_mask, token_type_ids
/// 3. Embedding lookup: word + position + token_type
/// 4. Apply LayerNorm to embeddings
/// 5. Run transformer encoder layers (6 layers for MiniLM)
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
            model_id: ModelId::Graph,
            message: format!("GraphModel tokenization failed: {}", e),
        })?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<f32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as f32)
        .collect();

    // Truncate to max_position_embeddings if needed
    let max_len = config.max_position_embeddings.min(GRAPH_MAX_TOKENS);
    let seq_len = token_ids.len().min(max_len);
    let token_ids = &token_ids[..seq_len];
    let attention_mask = &attention_mask[..seq_len];

    // Create GPU tensors
    let input_ids = Tensor::from_slice(token_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("GraphModel input_ids tensor failed: {}", e),
        }
    })?;

    let attention_mask_tensor =
        Tensor::from_slice(attention_mask, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("GraphModel attention_mask tensor failed: {}", e),
            }
        })?;

    // Token type IDs (all zeros for single sentence)
    let token_type_ids: Vec<u32> = vec![0u32; seq_len];
    let token_type_tensor =
        Tensor::from_slice(&token_type_ids, (1, seq_len), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("GraphModel token_type tensor failed: {}", e),
            }
        })?;

    // Position IDs
    let position_ids: Vec<u32> = (0..seq_len as u32).collect();
    let position_tensor = Tensor::from_slice(&position_ids, (1, seq_len), device).map_err(|e| {
        EmbeddingError::GpuError {
            message: format!("GraphModel position_ids tensor failed: {}", e),
        }
    })?;

    // === EMBEDDING LAYER ===
    let embeddings = compute_embeddings(
        &input_ids,
        &position_tensor,
        &token_type_tensor,
        weights,
        seq_len,
        config.hidden_size,
    )?;

    // Apply LayerNorm to embeddings
    let embeddings = layer_norm(
        &embeddings,
        &weights.embeddings.layer_norm_weight,
        &weights.embeddings.layer_norm_bias,
        config.layer_norm_eps,
    )?;

    // === ENCODER LAYERS ===
    let mut hidden_states = embeddings;

    // Create attention mask for broadcasting: [batch, 1, 1, seq_len]
    let extended_attention_mask = create_extended_attention_mask(&attention_mask_tensor)?;

    for (layer_idx, layer) in weights.encoder_layers.iter().enumerate() {
        hidden_states = encoder_layer_forward(
            &hidden_states,
            layer,
            &extended_attention_mask,
            config,
            layer_idx,
        )?;
    }

    // === POOLING ===
    let pooled = mean_pooling(
        &hidden_states,
        &attention_mask_tensor,
        seq_len,
        config.hidden_size,
    )?;

    // L2 normalize
    let normalized = normalize_gpu(&pooled).map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel L2 normalize failed: {}", e),
    })?;

    // Convert to Vec<f32>
    let result: Vec<f32> = normalized
        .flatten_all()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel flatten output failed: {}", e),
        })?
        .to_vec1()
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel to_vec1 failed: {}", e),
        })?;

    Ok(result)
}

/// Compute embeddings from input tensors.
fn compute_embeddings(
    input_ids: &Tensor,
    position_tensor: &Tensor,
    token_type_tensor: &Tensor,
    weights: &BertWeights,
    seq_len: usize,
    hidden_size: usize,
) -> EmbeddingResult<Tensor> {
    let word_embeds = weights
        .embeddings
        .word_embeddings
        .index_select(
            &input_ids
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("GraphModel flatten input_ids failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel word embedding lookup failed: {}", e),
        })?
        .reshape((1, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel word embedding reshape failed: {}", e),
        })?;

    let position_embeds = weights
        .embeddings
        .position_embeddings
        .index_select(
            &position_tensor
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("GraphModel flatten position_ids failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel position embedding lookup failed: {}", e),
        })?
        .reshape((1, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel position embedding reshape failed: {}", e),
        })?;

    let token_type_embeds = weights
        .embeddings
        .token_type_embeddings
        .index_select(
            &token_type_tensor
                .flatten_all()
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("GraphModel flatten token_type_ids failed: {}", e),
                })?,
            0,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel token_type embedding lookup failed: {}", e),
        })?
        .reshape((1, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel token_type embedding reshape failed: {}", e),
        })?;

    // Sum embeddings
    ((word_embeds + position_embeds).map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel embedding add 1 failed: {}", e),
    })? + token_type_embeds)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel embedding add 2 failed: {}", e),
        })
}

/// Create extended attention mask for broadcasting.
fn create_extended_attention_mask(attention_mask_tensor: &Tensor) -> EmbeddingResult<Tensor> {
    let extended = attention_mask_tensor
        .unsqueeze(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel attention mask unsqueeze 1 failed: {}", e),
        })?
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel attention mask unsqueeze 2 failed: {}", e),
        })?;

    // Convert mask: 1.0 -> 0.0, 0.0 -> -10000.0
    let extended = ((extended * (-1.0)).map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel attention mask mul failed: {}", e),
    })? + 1.0)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel attention mask add failed: {}", e),
        })?
        * (-10000.0f64);

    extended.map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel attention mask scale failed: {}", e),
    })
}

/// Perform mean pooling over the sequence dimension.
fn mean_pooling(
    hidden_states: &Tensor,
    attention_mask_tensor: &Tensor,
    seq_len: usize,
    hidden_size: usize,
) -> EmbeddingResult<Tensor> {
    let mask_expanded = attention_mask_tensor
        .unsqueeze(2)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel mask expand failed: {}", e),
        })?
        .broadcast_as((1, seq_len, hidden_size))
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel mask broadcast failed: {}", e),
        })?;

    let masked_hidden = (hidden_states * mask_expanded).map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel masked multiply failed: {}", e),
    })?;

    let sum_hidden = masked_hidden.sum(1).map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel sum hidden failed: {}", e),
    })?;

    let mask_sum = attention_mask_tensor
        .sum(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel mask sum failed: {}", e),
        })?
        .unsqueeze(1)
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel mask sum unsqueeze failed: {}", e),
        })?
        .broadcast_as(sum_hidden.shape())
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel mask sum broadcast failed: {}", e),
        })?;

    (sum_hidden
        / (mask_sum + 1e-9f64).map_err(|e| EmbeddingError::GpuError {
            message: format!("GraphModel mask sum add eps failed: {}", e),
        })?)
    .map_err(|e| EmbeddingError::GpuError {
        message: format!("GraphModel mean pooling div failed: {}", e),
    })
}
