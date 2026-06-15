//! Error types for clustering operations.

use thiserror::Error;

use crate::error::StorageError;
use crate::teleological::Embedder;

/// Errors that can occur during clustering operations.
#[derive(Debug, Error)]
pub enum ClusterError {
    /// Not enough data points for clustering.
    #[error("Insufficient data: required {required}, actual {actual}")]
    InsufficientData {
        /// Minimum required data points
        required: usize,
        /// Actual data points provided
        actual: usize,
    },

    /// Embedding dimension doesn't match expected dimension for space.
    #[error("Dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch {
        /// Expected dimension for this embedding space
        expected: usize,
        /// Actual dimension provided
        actual: usize,
    },

    /// No valid clusters found (all points are noise).
    #[error("No valid clusters found")]
    NoValidClusters,

    /// Storage operation failed.
    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),

    /// Invalid parameter provided.
    #[error("Invalid parameter: {message}")]
    InvalidParameter {
        /// Description of what's wrong with the parameter
        message: String,
    },

    /// Embedding space not initialized for clustering.
    #[error("Space not initialized: {0:?}")]
    SpaceNotInitialized(Embedder),
}

impl ClusterError {
    /// Create an InsufficientData error.
    pub fn insufficient_data(required: usize, actual: usize) -> Self {
        Self::InsufficientData { required, actual }
    }

    /// Create a DimensionMismatch error.
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an InvalidParameter error.
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_variants_are_debug() {
        // Comprehensive test: all variants implement Debug and Display with correct messages
        let errors: Vec<ClusterError> = vec![
            ClusterError::insufficient_data(3, 1),
            ClusterError::dimension_mismatch(1024, 512),
            ClusterError::NoValidClusters,
            ClusterError::invalid_parameter("min_cluster_size must be > 0"),
            ClusterError::SpaceNotInitialized(Embedder::Semantic),
        ];

        let expected_substrings = [
            "required 3",
            "expected 1024",
            "No valid clusters",
            "min_cluster_size",
            "Semantic",
        ];

        for (err, expected) in errors.iter().zip(expected_substrings.iter()) {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty(), "Debug should produce output");
            let display = err.to_string();
            assert!(
                display.contains(expected),
                "Display for {:?} should contain '{}', got: {}",
                err,
                expected,
                display
            );
        }

        println!("[PASS] test_error_variants_are_debug - all variants implement Debug+Display");
    }
}
