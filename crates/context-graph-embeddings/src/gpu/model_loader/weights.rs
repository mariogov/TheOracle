//! Weight structures for BERT model components.
//!
//! Defines the tensor storage for all BERT architecture components:
//! - Embeddings (word, position, token_type, LayerNorm)
//! - Self-attention (Q, K, V projections and output)
//! - Feed-forward networks (intermediate and output projections)
//! - Pooler (optional CLS token projection)

use candle_core::{Device, Tensor};

use super::config::BertConfig;

/// Embedding weights (word, position, token_type, LayerNorm).
#[derive(Debug)]
pub struct EmbeddingWeights {
    /// Word embeddings: [vocab_size, hidden_size]
    pub word_embeddings: Tensor,
    /// Position embeddings: [max_position, hidden_size]
    pub position_embeddings: Tensor,
    /// Token type embeddings: [type_vocab_size, hidden_size]
    pub token_type_embeddings: Tensor,
    /// LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Self-attention weights for a single layer.
#[derive(Debug)]
pub struct AttentionWeights {
    /// Query projection: [hidden_size, hidden_size]
    pub query_weight: Tensor,
    /// Query bias: [hidden_size]
    pub query_bias: Tensor,
    /// Key projection: [hidden_size, hidden_size]
    pub key_weight: Tensor,
    /// Key bias: [hidden_size]
    pub key_bias: Tensor,
    /// Value projection: [hidden_size, hidden_size]
    pub value_weight: Tensor,
    /// Value bias: [hidden_size]
    pub value_bias: Tensor,
    /// Output projection: [hidden_size, hidden_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Attention output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Attention output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Feed-forward network weights for a single layer.
#[derive(Debug)]
pub struct FfnWeights {
    /// Intermediate (up) projection: [hidden_size, intermediate_size]
    pub intermediate_weight: Tensor,
    /// Intermediate bias: [intermediate_size]
    pub intermediate_bias: Tensor,
    /// Output (down) projection: [intermediate_size, hidden_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Complete weights for a single encoder layer.
#[derive(Debug)]
pub struct EncoderLayerWeights {
    /// Self-attention weights.
    pub attention: AttentionWeights,
    /// Feed-forward network weights.
    pub ffn: FfnWeights,
}

/// Pooler weights for [CLS] token projection.
#[derive(Debug)]
pub struct PoolerWeights {
    /// Dense projection: [hidden_size, hidden_size]
    pub dense_weight: Tensor,
    /// Dense bias: [hidden_size]
    pub dense_bias: Tensor,
}

/// Complete BERT model weights loaded from safetensors.
#[derive(Debug)]
pub struct BertWeights {
    /// Model configuration.
    pub config: BertConfig,
    /// Embedding layer weights.
    pub embeddings: EmbeddingWeights,
    /// Encoder layer weights (one per layer).
    pub encoder_layers: Vec<EncoderLayerWeights>,
    /// Pooler weights (optional, may not exist in some models).
    pub pooler: Option<PoolerWeights>,
    /// Device the weights are loaded on.
    pub(crate) device: &'static Device,
}

impl BertWeights {
    /// Get the device these weights are loaded on.
    pub fn device(&self) -> &'static Device {
        self.device
    }

    /// Get total parameter count.
    pub fn param_count(&self) -> usize {
        let embedding_params = self.embeddings.word_embeddings.elem_count()
            + self.embeddings.position_embeddings.elem_count()
            + self.embeddings.token_type_embeddings.elem_count()
            + self.embeddings.layer_norm_weight.elem_count()
            + self.embeddings.layer_norm_bias.elem_count();

        let layer_params: usize = self
            .encoder_layers
            .iter()
            .map(|layer| {
                layer.attention.query_weight.elem_count()
                    + layer.attention.query_bias.elem_count()
                    + layer.attention.key_weight.elem_count()
                    + layer.attention.key_bias.elem_count()
                    + layer.attention.value_weight.elem_count()
                    + layer.attention.value_bias.elem_count()
                    + layer.attention.output_weight.elem_count()
                    + layer.attention.output_bias.elem_count()
                    + layer.attention.layer_norm_weight.elem_count()
                    + layer.attention.layer_norm_bias.elem_count()
                    + layer.ffn.intermediate_weight.elem_count()
                    + layer.ffn.intermediate_bias.elem_count()
                    + layer.ffn.output_weight.elem_count()
                    + layer.ffn.output_bias.elem_count()
                    + layer.ffn.layer_norm_weight.elem_count()
                    + layer.ffn.layer_norm_bias.elem_count()
            })
            .sum();

        let pooler_params = self
            .pooler
            .as_ref()
            .map(|p| p.dense_weight.elem_count() + p.dense_bias.elem_count())
            .unwrap_or(0);

        embedding_params + layer_params + pooler_params
    }

    /// Get estimated VRAM usage in bytes (F32).
    pub fn vram_bytes(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }
}
