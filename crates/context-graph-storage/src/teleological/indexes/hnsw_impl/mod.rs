//! HNSW index implementation using usearch for O(log n) graph traversal.
//!
//! TASK-STORAGE-P1-001: Replaced brute force O(n) linear scan with
//! production-grade HNSW via usearch crate.
//!
//! Each HnswEmbedderIndex wraps usearch::Index with configuration from HnswConfig.
//!
//! # FAIL FAST
//!
//! - Wrong dimension: `IndexError::DimensionMismatch`
//! - NaN/Inf in vector: `IndexError::InvalidVector`
//! - E6/E12/E13 on HnswEmbedderIndex::new(): `panic!` with clear message
//! - usearch operation failure: `IndexError::OperationFailed`
//!
//! # Module Structure
//!
//! - `types` - Core struct definition and helper functions
//! - `ops` - EmbedderIndexOps trait implementation
//! - `tests` - Comprehensive test suite

mod ops;
mod types;

#[cfg(test)]
mod tests;

// Re-export main types for backwards compatibility
pub use self::types::HnswEmbedderIndex;
