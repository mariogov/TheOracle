//! Constants for the semantic embedding model.
//!
//! These define the model configuration for intfloat/e5-large-v2.

/// Native dimension for e5-large-v2 model.
pub const SEMANTIC_DIMENSION: usize = 1024;

/// Maximum tokens for e5-large-v2 model.
pub const SEMANTIC_MAX_TOKENS: usize = 512;

/// Latency budget in milliseconds (P95 target).
pub const SEMANTIC_LATENCY_BUDGET_MS: u32 = 5;

/// Instruction prefix for query mode (search queries).
pub const QUERY_PREFIX: &str = "query: ";

/// Instruction prefix for passage mode (documents) - DEFAULT.
pub const PASSAGE_PREFIX: &str = "passage: ";
