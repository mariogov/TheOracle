//! Configuration types for Qwen2 model (Qodo-Embed-1-1.5B).

use std::path::Path;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

/// Qwen2 configuration parsed from config.json.
#[derive(Debug, Clone)]
pub struct QwenConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden layer size.
    pub hidden_size: usize,
    /// Intermediate FFN size.
    pub intermediate_size: usize,
    /// Number of hidden layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE theta for position encoding.
    pub rope_theta: f64,
    /// Maximum position embeddings.
    #[allow(dead_code)]
    pub max_position_embeddings: usize,
    /// Head dimension (computed from hidden_size / num_attention_heads).
    pub head_dim: usize,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151646,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 131072,
            head_dim: 128, // 1536 / 12 = 128
        }
    }
}

impl QwenConfig {
    /// Load config from JSON file.
    pub fn from_path(model_path: &Path) -> EmbeddingResult<Self> {
        let config_path = model_path.join("config.json");
        let config_content =
            std::fs::read_to_string(&config_path).map_err(|e| EmbeddingError::ModelLoadError {
                model_id: ModelId::Code,
                source: Box::new(e),
            })?;

        #[derive(serde::Deserialize)]
        struct RawConfig {
            vocab_size: usize,
            hidden_size: usize,
            intermediate_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            #[serde(default = "default_num_kv_heads")]
            num_key_value_heads: usize,
            #[serde(default = "default_rms_norm_eps")]
            rms_norm_eps: f64,
            #[serde(default = "default_rope_theta")]
            rope_theta: f64,
            #[serde(default = "default_max_position")]
            max_position_embeddings: usize,
        }

        fn default_num_kv_heads() -> usize {
            2
        }
        fn default_rms_norm_eps() -> f64 {
            1e-6
        }
        fn default_rope_theta() -> f64 {
            1_000_000.0
        }
        fn default_max_position() -> usize {
            131072
        }

        let raw: RawConfig =
            serde_json::from_str(&config_content).map_err(|e| EmbeddingError::ConfigError {
                message: format!("Qwen2 config parse failed: {}", e),
            })?;

        let head_dim = raw.hidden_size / raw.num_attention_heads;

        Ok(QwenConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            max_position_embeddings: raw.max_position_embeddings,
            head_dim,
        })
    }
}
