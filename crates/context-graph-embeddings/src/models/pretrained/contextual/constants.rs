//! Constants for ContextualModel (intfloat/e5-base-v2).
//!
//! # Model Specification
//!
//! - Architecture: BERT-base (12 layers)
//! - Model: intfloat/e5-base-v2
//! - Training: Trained for asymmetric retrieval with query/passage prefixes
//! - Output: 768D dense embedding optimized for asymmetric search
//! - Asymmetry: "query: " prefix for intents, "passage: " prefix for context

/// Output dimension for contextual embeddings.
pub const CONTEXTUAL_DIMENSION: usize = 768;

/// Maximum sequence length (from config.json max_position_embeddings).
pub const CONTEXTUAL_MAX_TOKENS: usize = 512;

/// Model name for logging and identification.
pub const CONTEXTUAL_MODEL_NAME: &str = "intfloat/e5-base-v2";

/// Latency budget for single embedding (milliseconds).
/// Slightly higher than before to account for dual-pass encoding.
pub const CONTEXTUAL_LATENCY_BUDGET_MS: u64 = 20;

/// Vocabulary size for BERT tokenizer.
pub const CONTEXTUAL_VOCAB_SIZE: usize = 30522;

/// Number of hidden layers in BERT encoder.
pub const CONTEXTUAL_NUM_LAYERS: usize = 12;

/// Number of attention heads per layer.
pub const CONTEXTUAL_NUM_HEADS: usize = 12;

/// Hidden size (matches CONTEXTUAL_DIMENSION for E5-base).
pub const CONTEXTUAL_HIDDEN_SIZE: usize = 768;

/// Intermediate FFN size.
pub const CONTEXTUAL_INTERMEDIATE_SIZE: usize = 3072;

/// Layer norm epsilon.
pub const CONTEXTUAL_LAYER_NORM_EPS: f64 = 1e-12;

// =============================================================================
// E5-base-v2 Asymmetric Search Prefixes
// =============================================================================
//
// E5 models use prefix-based encoding for asymmetric retrieval:
// - "query: " prefix for search queries (intent)
// - "passage: " prefix for documents/passages (context)
//
// This creates genuinely learned asymmetric representations without
// requiring separate projection matrices.

/// Prefix for intent/query embeddings.
/// E5-base-v2 uses "query: " prefix for search queries.
pub const INTENT_PREFIX: &str = "query: ";

/// Prefix for context/passage embeddings.
/// E5-base-v2 uses "passage: " prefix for documents/passages.
pub const CONTEXT_PREFIX: &str = "passage: ";
