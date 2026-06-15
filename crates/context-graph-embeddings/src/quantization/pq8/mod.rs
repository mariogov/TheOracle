//! Product Quantization (PQ-8) Encoder Implementation
//!
//! Implements Product Quantization with 8 subvectors and 256 centroids per subvector.
//! Used for E1_Semantic, E5_Causal, E7_Code, E10_Multimodal embedders.
//!
//! # Constitution Alignment
//!
//! - Compression: 32x (e.g., 1024D f32 -> 8 bytes)
//! - Max Recall Loss: <5%
//! - Used for: E1, E5, E7, E10
//!
//! # Algorithm
//!
//! 1. Split embedding into 8 subvectors of dimension D/8
//! 2. For each subvector, find the nearest centroid (1 of 256)
//! 3. Store 8 centroid indices (1 byte each) = 8 bytes total
//!
//! # Codebook Management
//!
//! The encoder uses a default codebook initialized with uniformly spaced centroids.
//! For production use, train codebooks on actual embedding data using `train_codebook()`.
//!
//! # Module Organization
//!
//! - `types` - Constants, error types, and configuration
//! - `encoder` - PQ8Encoder struct and core quantization/dequantization
//! - `training` - Codebook training (k-means)
//! - `persistence` - Codebook save/load to binary files
//! - `tests` - Comprehensive test suite

pub mod encoder;
mod persistence;
pub mod training;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use encoder::PQ8Encoder;
pub use training::generate_realistic_embeddings;
pub use types::{
    KMeansConfig,
    PQ8QuantizationError,
    // Internal constants also re-exported for completeness
    CODEBOOK_MAGIC,
    CODEBOOK_VERSION,
    NUM_CENTROIDS,
    NUM_SUBVECTORS,
};
