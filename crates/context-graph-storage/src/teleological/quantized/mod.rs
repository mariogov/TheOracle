//! Quantized fingerprint storage for per-embedder HNSW indexing.
//!
//! TASK-EMB-022: Integrates `StoredQuantizedFingerprint` from context-graph-embeddings
//! into RocksDB storage with 13 dedicated column families (emb_0 through emb_12).
//!
//! # Architecture
//!
//! ```text
//! StoredQuantizedFingerprint (~17KB)
//! ├── Metadata (id, version, topic_profile, timestamps)
//! └── embeddings: HashMap<u8, QuantizedEmbedding>
//!     ├── emb_0: E1_Semantic (PQ-8, ~8 bytes)
//!     ├── emb_1: E2_TemporalRecent (Float8, ~512 bytes)
//!     ├── ...
//!     └── emb_12: E13_SPLADE (Sparse, ~2KB)
//!
//! RocksDB Column Families:
//! ├── fingerprints: Full StoredQuantizedFingerprint (metadata only mode)
//! ├── emb_0: Per-UUID QuantizedEmbedding for embedder 0
//! ├── emb_1: Per-UUID QuantizedEmbedding for embedder 1
//! ├── ...
//! └── emb_12: Per-UUID QuantizedEmbedding for embedder 12
//! ```
//!
//! # FAIL FAST Policy
//!
//! **NO FALLBACKS. NO WORKAROUNDS.**
//!
//! - Missing embedder → panic with full context
//! - Serialization error → panic with full context
//! - Column family missing → panic with full context
//! - All 13 embedders MUST be present on store/load
//!
//! # Storage Size Targets
//!
//! - Per fingerprint: ~17KB total (Constitution requirement)
//! - Per embedder: ~1-2KB average (varies by quantization method)

mod error;
mod helpers;
mod rocksdb_impl;
#[cfg(test)]
mod tests;
mod trait_def;

// Re-export all public items for backwards compatibility
pub use self::error::{QuantizedStorageError, QuantizedStorageResult};
pub use self::helpers::{
    deserialize_quantized_embedding, embedder_key, serialize_quantized_embedding,
};
pub use self::trait_def::QuantizedFingerprintStorage;
