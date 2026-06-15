//! Multi-Array Embedding Storage for Teleological Fingerprints.
//!
//! This module stores all 13 embeddings (E1-E12 + E13 SPLADE) as SEPARATE
//! arrays. Each embedding maintains its native dimension for per-space indexing.
//! The 13-embedding array IS the teleological vector (Royse 2026).
//!
//! # Architecture: Multi-Array Storage
//!
//! ```text
//! Individual Models (E1-E13)
//!          ↓
//!     ModelEmbedding (per model, native dimension)
//!          ↓
//!     MultiArrayEmbedding (this module) ← collects all 13 SEPARATELY
//!          ↓
//!     Multi-Array Storage (all 13 embeddings preserved as separate arrays)
//!          ↓
//!     Per-Space Indexes (13 HNSW indexes, one per embedding space)
//! ```
//!
//! # Key Design Principles
//!
//! - **100% Information Preserved**: All embeddings stored at native dimensions
//! - **Per-Space Indexing**: Each embedding space has its own HNSW index
//! - **Different Similarity Semantics**: E5 uses asymmetric similarity, E12 uses MaxSim
//! - **5-Stage Pipeline**: Different stages use different embeddings
//! - **RRF Fusion**: Scores are fused via Reciprocal Rank Fusion, not vector operations
//!
//! # Example
//!
//! ```
//! use context_graph_embeddings::types::{MultiArrayEmbedding, ModelEmbedding, ModelId};
//!
//! let mut multi = MultiArrayEmbedding::new();
//! assert_eq!(multi.filled_count(), 0);
//!
//! // Add one embedding (stored separately in its native dimension)
//! let model_id = ModelId::Semantic;
//! let dim = model_id.projected_dimension();
//! let mut emb = ModelEmbedding::new(model_id, vec![0.1; dim], 100);
//! emb.set_projected(true);
//! multi.set(emb);
//!
//! assert_eq!(multi.filled_count(), 1);
//! assert!(!multi.is_complete()); // Need all 12
//! ```

mod core;
mod operations;

#[cfg(test)]
mod tests;

pub use self::core::MultiArrayEmbedding;
