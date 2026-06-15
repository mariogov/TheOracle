//! Error types for dense vector similarity computation.

use thiserror::Error;

/// Errors from dense vector similarity computation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DenseSimilarityError {
    /// Dimension mismatch between vectors.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension (from first vector)
        expected: usize,
        /// Actual dimension (from second vector)
        actual: usize,
    },

    /// Empty vector provided.
    #[error("Empty vector provided")]
    EmptyVector,

    /// Zero magnitude vector - cosine undefined.
    #[error("Zero magnitude vector - cosine undefined")]
    ZeroMagnitude,
}
