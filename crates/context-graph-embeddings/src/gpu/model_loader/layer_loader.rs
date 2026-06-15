//! Layer-level weight loading functions for BERT and MPNet encoder layers.
//!
//! Provides functions for loading attention and FFN weights for each encoder layer.
//! Supports both BERT naming (self.query, self.key, self.value) and MPNet naming
//! (attn.q, attn.k, attn.v) for compatibility with both model architectures.

use candle_nn::VarBuilder;
use std::path::Path;

use super::config::BertConfig;
use super::error::ModelLoadError;
use super::tensor_utils::get_tensor;
use super::weights::{AttentionWeights, EncoderLayerWeights, FfnWeights};

/// Load a single encoder layer with optional model prefix.
pub fn load_encoder_layer(
    vb: &VarBuilder,
    config: &BertConfig,
    layer_idx: usize,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<EncoderLayerWeights, ModelLoadError> {
    let attention = load_attention_weights(vb, config, layer_idx, model_dir, model_prefix)?;
    let ffn = load_ffn_weights(vb, config, layer_idx, model_dir, model_prefix)?;

    Ok(EncoderLayerWeights { attention, ffn })
}

/// Load self-attention weights for a layer with optional model prefix.
pub fn load_attention_weights(
    vb: &VarBuilder,
    config: &BertConfig,
    layer_idx: usize,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<AttentionWeights, ModelLoadError> {
    let prefix = format!("{}encoder.layer.{}.attention", model_prefix, layer_idx);
    let model_path = model_dir.display().to_string();

    // Self attention Q/K/V
    let query_weight = get_tensor(
        vb,
        &format!("{}.self.query.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let query_bias = get_tensor(
        vb,
        &format!("{}.self.query.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    let key_weight = get_tensor(
        vb,
        &format!("{}.self.key.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let key_bias = get_tensor(
        vb,
        &format!("{}.self.key.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    let value_weight = get_tensor(
        vb,
        &format!("{}.self.value.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let value_bias = get_tensor(
        vb,
        &format!("{}.self.value.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    // Output projection
    let output_weight = get_tensor(
        vb,
        &format!("{}.output.dense.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let output_bias = get_tensor(
        vb,
        &format!("{}.output.dense.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    // LayerNorm
    let layer_norm_weight = get_tensor(
        vb,
        &format!("{}.output.LayerNorm.weight", prefix),
        &[config.hidden_size],
        &model_path,
    )?;
    let layer_norm_bias = get_tensor(
        vb,
        &format!("{}.output.LayerNorm.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    Ok(AttentionWeights {
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        layer_norm_weight,
        layer_norm_bias,
    })
}

/// Load feed-forward network weights for a layer with optional model prefix.
pub fn load_ffn_weights(
    vb: &VarBuilder,
    config: &BertConfig,
    layer_idx: usize,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<FfnWeights, ModelLoadError> {
    let prefix = format!("{}encoder.layer.{}", model_prefix, layer_idx);
    let model_path = model_dir.display().to_string();

    // Intermediate (up projection)
    let intermediate_weight = get_tensor(
        vb,
        &format!("{}.intermediate.dense.weight", prefix),
        &[config.intermediate_size, config.hidden_size],
        &model_path,
    )?;
    let intermediate_bias = get_tensor(
        vb,
        &format!("{}.intermediate.dense.bias", prefix),
        &[config.intermediate_size],
        &model_path,
    )?;

    // Output (down projection)
    let output_weight = get_tensor(
        vb,
        &format!("{}.output.dense.weight", prefix),
        &[config.hidden_size, config.intermediate_size],
        &model_path,
    )?;
    let output_bias = get_tensor(
        vb,
        &format!("{}.output.dense.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    // LayerNorm
    let layer_norm_weight = get_tensor(
        vb,
        &format!("{}.output.LayerNorm.weight", prefix),
        &[config.hidden_size],
        &model_path,
    )?;
    let layer_norm_bias = get_tensor(
        vb,
        &format!("{}.output.LayerNorm.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    Ok(FfnWeights {
        intermediate_weight,
        intermediate_bias,
        output_weight,
        output_bias,
        layer_norm_weight,
        layer_norm_bias,
    })
}

// ===========================================================================
// MPNet-specific loading functions
//
// MPNet uses different weight naming than BERT:
// - BERT: encoder.layer.X.attention.self.query.weight
// - MPNet: encoder.layer.X.attention.attn.q.weight
//
// MPNet also has LayerNorm in different positions:
// - BERT: attention.output.LayerNorm, output.LayerNorm
// - MPNet: attention.LayerNorm, output.LayerNorm
// ===========================================================================

/// Load a single encoder layer for MPNet architecture.
///
/// MPNet uses different weight naming than BERT.
pub fn load_mpnet_encoder_layer(
    vb: &VarBuilder,
    config: &BertConfig,
    layer_idx: usize,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<EncoderLayerWeights, ModelLoadError> {
    let attention = load_mpnet_attention_weights(vb, config, layer_idx, model_dir, model_prefix)?;
    let ffn = load_mpnet_ffn_weights(vb, config, layer_idx, model_dir, model_prefix)?;

    Ok(EncoderLayerWeights { attention, ffn })
}

/// Load self-attention weights for MPNet architecture.
///
/// MPNet uses attn.q, attn.k, attn.v, attn.o instead of BERT's
/// self.query, self.key, self.value, output.dense.
fn load_mpnet_attention_weights(
    vb: &VarBuilder,
    config: &BertConfig,
    layer_idx: usize,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<AttentionWeights, ModelLoadError> {
    let prefix = format!("{}encoder.layer.{}.attention", model_prefix, layer_idx);
    let model_path = model_dir.display().to_string();

    // MPNet self attention Q/K/V/O (different naming than BERT)
    let query_weight = get_tensor(
        vb,
        &format!("{}.attn.q.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let query_bias = get_tensor(
        vb,
        &format!("{}.attn.q.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    let key_weight = get_tensor(
        vb,
        &format!("{}.attn.k.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let key_bias = get_tensor(
        vb,
        &format!("{}.attn.k.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    let value_weight = get_tensor(
        vb,
        &format!("{}.attn.v.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let value_bias = get_tensor(
        vb,
        &format!("{}.attn.v.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    // Output projection (attn.o in MPNet)
    let output_weight = get_tensor(
        vb,
        &format!("{}.attn.o.weight", prefix),
        &[config.hidden_size, config.hidden_size],
        &model_path,
    )?;
    let output_bias = get_tensor(
        vb,
        &format!("{}.attn.o.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    // LayerNorm (MPNet has it directly under attention, not attention.output)
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

    Ok(AttentionWeights {
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        layer_norm_weight,
        layer_norm_bias,
    })
}

/// Load feed-forward network weights for MPNet architecture.
///
/// MPNet FFN structure is similar to BERT but LayerNorm location may differ.
fn load_mpnet_ffn_weights(
    vb: &VarBuilder,
    config: &BertConfig,
    layer_idx: usize,
    model_dir: &Path,
    model_prefix: &str,
) -> Result<FfnWeights, ModelLoadError> {
    let prefix = format!("{}encoder.layer.{}", model_prefix, layer_idx);
    let model_path = model_dir.display().to_string();

    // Intermediate (up projection)
    let intermediate_weight = get_tensor(
        vb,
        &format!("{}.intermediate.dense.weight", prefix),
        &[config.intermediate_size, config.hidden_size],
        &model_path,
    )?;
    let intermediate_bias = get_tensor(
        vb,
        &format!("{}.intermediate.dense.bias", prefix),
        &[config.intermediate_size],
        &model_path,
    )?;

    // Output (down projection)
    let output_weight = get_tensor(
        vb,
        &format!("{}.output.dense.weight", prefix),
        &[config.hidden_size, config.intermediate_size],
        &model_path,
    )?;
    let output_bias = get_tensor(
        vb,
        &format!("{}.output.dense.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    // LayerNorm
    let layer_norm_weight = get_tensor(
        vb,
        &format!("{}.output.LayerNorm.weight", prefix),
        &[config.hidden_size],
        &model_path,
    )?;
    let layer_norm_bias = get_tensor(
        vb,
        &format!("{}.output.LayerNorm.bias", prefix),
        &[config.hidden_size],
        &model_path,
    )?;

    Ok(FfnWeights {
        intermediate_weight,
        intermediate_bias,
        output_weight,
        output_bias,
        layer_norm_weight,
        layer_norm_bias,
    })
}
