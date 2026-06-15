//! EmbeddingSlice type for uniform access to different embedding representations.

use crate::types::fingerprint::SparseVector;

/// Reference type for returning embedding slices without copying.
///
/// This enum allows uniform access to all 13 embedding types while preserving
/// their different representations (dense, sparse, token-level).
#[derive(Debug)]
pub enum EmbeddingSlice<'a> {
    /// Dense embedding as a contiguous f32 slice.
    Dense(&'a [f32]),

    /// Sparse embedding (E6 SPLADE).
    Sparse(&'a SparseVector),

    /// Token-level embedding (E12 ColBERT) - variable number of 128D tokens.
    TokenLevel(&'a [Vec<f32>]),
}
