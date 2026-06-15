//! HnswMultiSpaceIndex implementation with 12 HNSW indexes using real hnsw_rs.
//!
//! Implements `MultiSpaceIndexManager` trait for the 5-stage retrieval pipeline.
//!
//! # CRITICAL: NO FALLBACKS
//!
//! This implementation uses the real hnsw_rs library. If any HNSW operation fails,
//! the system will ERROR OUT with detailed logging. No mock data, no fallbacks.
//!
//! # Index Architecture
//!
//! | Index Type | Count | Purpose | Stage |
//! |------------|-------|---------|-------|
//! | HNSW | 10 | E1-E5, E7-E11 dense | Stage 3 |
//! | HNSW | 1 | E1 Matryoshka 128D | Stage 2 |
//!
//! # Performance Requirements (constitution.yaml)
//!
//! - `add_vector()`: <1ms per index
//! - `search()`: <10ms per index
//! - `persist()`: <1s for 100K vectors
//!
//! # Module Structure
//!
//! - `types` - Type aliases and persistence data structures
//! - `real_hnsw` - RealHnswIndex implementation (core HNSW operations)
//! - `persistence` - Persistence operations (persist/load)
//! - `multi_space` - HnswMultiSpaceIndex struct and helpers
//! - `multi_space_trait` - MultiSpaceIndexManager trait implementation

mod multi_space;
mod multi_space_trait;
mod persistence;
mod real_hnsw;
mod types;

#[cfg(test)]
mod tests;

// Re-export public types for backwards compatibility
pub use multi_space::HnswMultiSpaceIndex;
pub use real_hnsw::RealHnswIndex;
pub use types::HnswPersistenceData;

// Re-export config types from parent module
pub use super::config::{DistanceMetric, EmbedderIndex, HnswConfig};
