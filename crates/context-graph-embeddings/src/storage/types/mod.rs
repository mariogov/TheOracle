//! Quantized storage types for per-embedder HNSW indexing.
//!
//! These types support the Constitution's 5-stage retrieval pipeline, specifically:
//! - Stage 3: Multi-space reranking with RRF fusion across 13 embedders
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**
//!
//! - All deserialization errors panic with full context
//! - No silent fallbacks or default values
//! - Version mismatches are fatal (no migration support)
//!
//! # Storage Size Targets (Constitution)
//!
//! - Unquantized TeleologicalFingerprint: ~63KB
//! - Quantized StoredQuantizedFingerprint: ~17KB (63% reduction)
//! - This 17KB target achieved via per-embedder quantization

pub mod constants;
pub mod fingerprint;
pub mod index_entry;
pub mod query_results;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::constants::*;
pub use self::fingerprint::StoredQuantizedFingerprint;
pub use self::index_entry::IndexEntry;
pub use self::query_results::{EmbedderQueryResult, MultiSpaceQueryResult};
