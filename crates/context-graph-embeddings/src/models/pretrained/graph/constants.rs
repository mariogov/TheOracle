//! Constants for the GraphModel (upgraded to e5-large-v2 for E8).
//!
//! These constants define the model's architecture and operational parameters.
//!
//! # E8 Dimension Change
//!
//! E8 has been upgraded from MiniLM (384D) to e5-large-v2 (1024D):
//! - Shares the model with E1 (no extra VRAM)
//! - Better semantic understanding for graph relationships
//! - Support asymmetric source/target embeddings via learned projections

/// Native dimension for e5-large-v2 graph embeddings (1024D).
///
/// Previously: MiniLM at 384D
/// Now: e5-large-v2 at 1024D (shared with E1)
pub const GRAPH_DIMENSION: usize = 1024;

/// Maximum tokens for e5-large-v2 (standard BERT-family limit).
pub const GRAPH_MAX_TOKENS: usize = 512;

/// Latency budget in milliseconds (P95 target).
pub const GRAPH_LATENCY_BUDGET_MS: u64 = 5;

/// HuggingFace model repository name.
///
/// Now uses e5-large-v2 (shared with E1) instead of MiniLM.
pub const GRAPH_MODEL_NAME: &str = "intfloat/e5-large-v2";

/// Maximum number of neighbor context entries for encode_context.
pub const MAX_CONTEXT_NEIGHBORS: usize = 5;
