//! PQ-8 types, constants, and error definitions.
//!
//! This module contains the core type definitions for PQ-8 quantization:
//! - Constants for PQ-8 algorithm parameters
//! - Error types for quantization operations
//! - Configuration types for k-means training

use std::fmt;

/// Number of subvectors for PQ-8.
pub const NUM_SUBVECTORS: usize = 8;

/// Number of centroids per subvector.
pub const NUM_CENTROIDS: usize = 256;

/// Magic bytes for codebook file format identification.
pub const CODEBOOK_MAGIC: &[u8; 4] = b"PQ8C";

/// Current codebook file format version.
pub const CODEBOOK_VERSION: u8 = 1;

/// Errors specific to PQ8 quantization operations.
#[derive(Debug, Clone)]
pub enum PQ8QuantizationError {
    /// Input embedding is empty.
    EmptyEmbedding,
    /// Input contains NaN values.
    ContainsNaN { index: usize },
    /// Input contains infinite values.
    ContainsInfinity { index: usize },
    /// Embedding dimension not divisible by 8.
    DimensionNotDivisible { dim: usize },
    /// Codebook dimension mismatch.
    CodebookDimensionMismatch { expected: usize, got: usize },
    /// Metadata type mismatch during dequantization.
    InvalidMetadata { expected: &'static str, got: String },
    /// Data length mismatch (should be 8 bytes).
    InvalidDataLength { expected: usize, got: usize },
    /// Insufficient training samples for codebook training.
    InsufficientSamples { required: usize, provided: usize },
    /// Sample dimension mismatch during training.
    SampleDimensionMismatch {
        sample_idx: usize,
        expected: usize,
        got: usize,
    },
    /// K-means clustering did not converge.
    KMeansDidNotConverge {
        iterations: usize,
        max_iterations: usize,
    },
    /// IO error during codebook persistence.
    IoError { message: String },
    /// Deserialization error during codebook loading.
    DeserializationError { message: String },
    /// Invalid codebook file format or version.
    InvalidCodebookFormat { message: String },
}

impl fmt::Display for PQ8QuantizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEmbedding => {
                write!(f, "Empty embedding: cannot quantize zero-length vector")
            }
            Self::ContainsNaN { index } => {
                write!(f, "Invalid input: NaN value at index {}", index)
            }
            Self::ContainsInfinity { index } => {
                write!(f, "Invalid input: Infinity value at index {}", index)
            }
            Self::DimensionNotDivisible { dim } => {
                write!(
                    f,
                    "Dimension {} not divisible by {} subvectors",
                    dim, NUM_SUBVECTORS
                )
            }
            Self::CodebookDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Codebook dimension mismatch: expected {}, got {}",
                    expected, got
                )
            }
            Self::InvalidMetadata { expected, got } => {
                write!(f, "Invalid metadata: expected {}, got {}", expected, got)
            }
            Self::InvalidDataLength { expected, got } => {
                write!(
                    f,
                    "Invalid data length: expected {} bytes, got {}",
                    expected, got
                )
            }
            Self::InsufficientSamples { required, provided } => {
                write!(
                    f,
                    "Insufficient training samples: required {} samples, got {}",
                    required, provided
                )
            }
            Self::SampleDimensionMismatch {
                sample_idx,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Sample {} dimension mismatch: expected {}, got {}",
                    sample_idx, expected, got
                )
            }
            Self::KMeansDidNotConverge {
                iterations,
                max_iterations,
            } => {
                write!(
                    f,
                    "K-means did not converge after {} iterations (max: {})",
                    iterations, max_iterations
                )
            }
            Self::IoError { message } => {
                write!(f, "IO error: {}", message)
            }
            Self::DeserializationError { message } => {
                write!(f, "Deserialization error: {}", message)
            }
            Self::InvalidCodebookFormat { message } => {
                write!(f, "Invalid codebook format: {}", message)
            }
        }
    }
}

impl std::error::Error for PQ8QuantizationError {}

/// Configuration for k-means codebook training.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Maximum number of k-means iterations.
    ///
    /// Typical values: 50-200. Higher values improve convergence but slow training.
    /// Default: 100
    pub max_iterations: usize,

    /// Convergence threshold (stop when centroid movement < threshold).
    ///
    /// Typical values: 1e-6 to 1e-4. Lower values give better accuracy but may
    /// require more iterations to converge.
    /// Default: 1e-6
    pub convergence_threshold: f32,

    /// Random seed for reproducible training.
    ///
    /// Used for k-means++ initialization. Same seed + same data = same codebook.
    /// Default: 42
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            seed: 42,
        }
    }
}

/// Simple deterministic RNG for reproducible k-means initialization.
/// Using a minimal LCG to avoid external dependencies in core quantization.
pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    /// Generate random f32 in range [0, 1).
    ///
    /// Uses 24 bits of entropy which matches f32 mantissa precision (23 bits + implicit 1).
    /// The shift by 40 bits extracts the upper 24 bits of the 64-bit state.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}
