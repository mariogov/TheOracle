//! Embedding layer weight loading for BERT models.
//!
//! Loads word, position, token_type embeddings and LayerNorm from safetensors.
//!
//! Note: token_type_embeddings is optional for models like MPNet that don't use them.
//! If not present in the model file, a zeros tensor is created instead.

use candle_nn::VarBuilder;
use std::path::Path;

use super::config::BertConfig;
use super::error::ModelLoadError;
use super::tensor_utils::{get_tensor, get_tensor_or_zeros};
use super::weights::{EmbeddingWeights, PoolerWeights};

/// Load embedding layer weights with optional model prefix.
///
/// # Token Type Embeddings
///
/// Some models like MPNet don't have token_type_embeddings. If not found in the
/// model file, a zeros tensor is created. This works because:
/// - MPNet always uses token_type_id=0 for all tokens
/// - Looking up index 0 in a zeros tensor returns all zeros
/// - Adding zeros to word+position embeddings has no effect
pub fn load_embeddings(
    vb: &VarBuilder,
    config: &BertConfig,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<EmbeddingWeights, ModelLoadError> {
    let prefix = if model_prefix.is_empty() {
        "embeddings".to_string()
    } else {
        format!("{}embeddings", model_prefix)
    };
    let model_path = model_dir.display().to_string();

    // Word embeddings: [vocab_size, hidden_size]
    let word_embeddings = get_tensor(
        vb,
        &format!("{}.word_embeddings.weight", prefix),
        &[config.vocab_size, config.hidden_size],
        &model_path,
    )?;

    // Get device from word_embeddings for creating optional tensors
    let device = word_embeddings.device().clone();

    // Position embeddings: [max_position_embeddings, hidden_size]
    let position_embeddings = get_tensor(
        vb,
        &format!("{}.position_embeddings.weight", prefix),
        &[config.max_position_embeddings, config.hidden_size],
        &model_path,
    )?;

    // Token type embeddings: [type_vocab_size, hidden_size]
    // OPTIONAL: MPNet and some other models don't have this.
    // If not present, we create a zeros tensor (since token_type_id=0 always).
    let token_type_embeddings = get_tensor_or_zeros(
        vb,
        &format!("{}.token_type_embeddings.weight", prefix),
        &[config.type_vocab_size, config.hidden_size],
        &device,
    )?;

    // LayerNorm
    let layer_norm_weight = get_tensor(
        vb,
        &format!("{}.LayerNorm.weight", prefix),
        &[config.hidden_size],
        &model_path,
    )?;
    let layer_norm_bias = get_tensor(
        vb,
        &format!("{}.LayerNorm.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    Ok(EmbeddingWeights {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        layer_norm_weight,
        layer_norm_bias,
    })
}

/// Load pooler weights with optional model prefix.
pub fn load_pooler(
    vb: &VarBuilder,
    config: &BertConfig,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<PoolerWeights, ModelLoadError> {
    let model_path = model_dir.display().to_string();

    let dense_weight = get_tensor(
        vb,
        &format!("{}pooler.dense.weight", model_prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let dense_bias = get_tensor(
        vb,
        &format!("{}pooler.dense.bias", model_prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    Ok(PoolerWeights {
        dense_weight,
        dense_bias,
    })
}
