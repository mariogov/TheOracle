//! Projection matrix for sparse-to-dense embedding conversion.
//!
//! This module defines the ProjectionMatrix struct for learned sparse projections.
//! The actual loading and projection logic are implemented in Logic Layer tasks.
//!
//! # Constitution Alignment
//! - E6_Sparse: "~30K 5%active" -> 1536D output via learned projection
//! - E13_SPLADE: Same architecture, same projection
//! - AP-007: No stub data in prod - hash fallback is FORBIDDEN
//!
//! # CRITICAL: No Fallback Policy
//! If the weight file is missing or invalid, the system MUST fail fast with a clear
//! error message. Under NO circumstances should the code fall back to hash-based
//! projection (`idx % projected_dim`). Such fallback violates Constitution AP-007.
//!
//! # Module Structure
//! - `types` - ProjectionMatrix struct and constants
//! - `impl_core` - Core implementation (load, project)
//! - `impl_batch` - Batch projection operations
//! - `error` - ProjectionError enum

mod error;
mod impl_batch;
mod impl_core;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public items for backwards compatibility
pub use self::types::*;

// Compile-time assertions to ensure constants match Constitution
const _: () = assert!(
    super::types::SPARSE_VOCAB_SIZE == 30522,
    "SPARSE_VOCAB_SIZE must be 30522 (BERT vocabulary)"
);

const _: () = assert!(
    super::types::SPARSE_PROJECTED_DIMENSION == 1536,
    "SPARSE_PROJECTED_DIMENSION must be 1536 per Constitution E6_Sparse"
);
