//! Graph edge storage for K-NN graphs and typed edges.
//!
//! This module provides persistent storage for the Knowledge Graph Linking
//! Enhancements feature:
//!
//! - **K-NN edges per embedder**: Each of 13 embedders maintains its own K-NN
//!   graph connecting memories to their nearest neighbors.
//! - **Typed edges**: Multi-relation edges derived from embedder agreement
//!   patterns, with 8 edge types based on which embedders agree.
//!
//! # Column Families
//!
//! - `embedder_edges`: K-NN edges per embedder
//!   - Key: `[embedder_id: u8][source_uuid: 16 bytes]` = 17 bytes
//!   - Value: Serialized `Vec<EmbedderEdge>` (k=20 neighbors)
//!
//! - `typed_edges`: Multi-relation typed edges
//!   - Key: `[source_uuid: 16 bytes][target_uuid: 16 bytes]` = 32 bytes
//!   - Value: Serialized `TypedEdge`
//!
//! - `typed_edges_by_type`: Secondary index for type-filtered queries
//!   - Key: `[edge_type: u8][source_uuid: 16 bytes]` = 17 bytes
//!   - Value: `target_uuid` (16 bytes)
//!
//! # Architecture Reference
//!
//! - ARCH-18: E5/E8 use asymmetric similarity (direction matters)
//! - AP-60: Temporal embedders (E2-E4) excluded from edge type detection
//! - AP-77: E5 MUST NOT use symmetric cosine
//!
//! # Usage
//!
//! ```ignore
//! use context_graph_storage::graph_edges::{EdgeRepository, GraphEdgeStorageResult};
//! use context_graph_core::graph_linking::EmbedderEdge;
//!
//! let repo = EdgeRepository::new(db);
//!
//! // Store K-NN edges for a node in E1 space
//! repo.store_embedder_edges(0, source_uuid, &edges)?;
//!
//! // Get neighbors
//! let neighbors = repo.get_embedder_edges(0, source_uuid)?;
//!
//! // Store a typed edge
//! repo.store_typed_edge(&typed_edge)?;
//! ```

mod builder;
mod repository;
mod serialization;
mod types;

// Re-export public types
pub use builder::{
    BackgroundGraphBuilder, BatchBuildResult, BuilderStats, GraphBuilderConfig, RebuildResult,
};
pub use repository::EdgeRepository;
pub use serialization::{
    deserialize_embedder_edges, deserialize_typed_edge, serialize_embedder_edges,
    serialize_typed_edge, GRAPH_EDGE_VERSION,
};
pub use types::{GraphEdgeStats, GraphEdgeStorageError, GraphEdgeStorageResult};
