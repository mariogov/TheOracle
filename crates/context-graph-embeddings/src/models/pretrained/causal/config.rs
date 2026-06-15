//! NomicBERT configuration for the causal embedding model (E5).
//!
//! This module contains the model configuration parsed from config.json
//! and the fixed constants for the nomic-embed-text-v1.5 causal model.

/// Native dimension for nomic-embed-text-v1.5.
pub const CAUSAL_DIMENSION: usize = 768;

/// Maximum tokens for causal encoding.
/// nomic-embed supports 8192, but causal sentences are typically 10-50 tokens.
/// 512 is sufficient and faster.
pub const CAUSAL_MAX_TOKENS: usize = 512;

/// Latency budget in milliseconds (P95 target).
pub const CAUSAL_LATENCY_BUDGET_MS: u32 = 8;

/// Instruction prefix for encoding text as a potential CAUSE.
///
/// Used by both `embed_as_cause()` and `gpu_forward_dual()` to ensure
/// consistent embeddings regardless of call path.
pub const CAUSE_INSTRUCTION: &str = "search_query: Identify the cause in: ";

/// Instruction prefix for encoding text as a potential EFFECT.
///
/// Used by both `embed_as_effect()` and `gpu_forward_dual()` to ensure
/// consistent embeddings regardless of call path.
pub const EFFECT_INSTRUCTION: &str = "search_query: Identify the effect of: ";

/// NomicBERT configuration parsed from config.json.
///
/// Based on nomic-ai/nomic-embed-text-v1.5:
/// - BERT-like architecture with rotary position embeddings
/// - SwiGLU activation in FFN
/// - Fused QKV projections (no separate Q/K/V weights)
/// - No biases in attention or FFN projections
#[derive(Debug, Clone)]
pub struct NomicConfig {
    /// Vocabulary size (30528 for nomic-embed).
    pub vocab_size: usize,
    /// Hidden layer size (768 for nomic-embed).
    pub hidden_size: usize,
    /// Number of hidden layers (12 for nomic-embed).
    pub num_hidden_layers: usize,
    /// Number of attention heads (12 for nomic-embed).
    pub num_attention_heads: usize,
    /// Intermediate FFN size (3072 for nomic-embed).
    pub intermediate_size: usize,
    /// Maximum position embeddings (8192 for nomic-embed).
    pub max_position_embeddings: usize,
    /// Token type vocabulary size (2 for nomic-embed).
    pub type_vocab_size: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
    /// Rotary embedding base frequency (1000 for nomic-embed).
    pub rotary_emb_base: f64,
    /// Fraction of head_dim that gets rotary embeddings (1.0 = all).
    pub rotary_emb_fraction: f64,
    /// Whether rotary embeddings use interleaved pairs.
    pub rotary_emb_interleaved: bool,
    /// Whether QKV projection has bias.
    pub qkv_proj_bias: bool,
    /// Whether MLP fc1 (gate/up) has bias.
    pub mlp_fc1_bias: bool,
    /// Whether MLP fc2 (down) has bias.
    pub mlp_fc2_bias: bool,
}

impl Default for NomicConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30528,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            rotary_emb_base: 1000.0,
            rotary_emb_fraction: 1.0,
            rotary_emb_interleaved: false,
            qkv_proj_bias: false,
            mlp_fc1_bias: false,
            mlp_fc2_bias: false,
        }
    }
}
