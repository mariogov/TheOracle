//! PQ-8 Encoder implementation.
//!
//! This module contains the `PQ8Encoder` struct which provides:
//! - Quantization of f32 embeddings to 8-byte PQ-8 format (32x compression)
//! - Dequantization back to approximate f32 values
//! - Thread-safe operation via Arc-wrapped codebook

use super::types::{PQ8QuantizationError, NUM_CENTROIDS, NUM_SUBVECTORS};
use crate::quantization::types::{
    PQ8Codebook, QuantizationMetadata, QuantizationMethod, QuantizedEmbedding,
};
use std::sync::Arc;
use tracing::{debug, warn};

/// PQ-8 Encoder for 32x compression of embedding vectors.
///
/// # Algorithm
///
/// 1. Split embedding into 8 subvectors
/// 2. Find nearest centroid for each subvector
/// 3. Store 8 centroid indices (1 byte each)
///
/// # Codebook
///
/// Uses a trained or default codebook. The default codebook provides reasonable
/// compression but may have higher recall loss than a properly trained codebook.
///
/// # Thread Safety
///
/// The encoder is thread-safe and can be shared across threads via Arc.
#[derive(Debug)]
pub struct PQ8Encoder {
    /// Codebook containing centroids for each subvector.
    codebook: Arc<PQ8Codebook>,
}

impl PQ8Encoder {
    /// Create a new PQ8 encoder with a default codebook for the given dimension.
    ///
    /// The default codebook uses uniformly spaced centroids in [-1, 1] range.
    /// For production use, train a codebook on actual embedding data.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - The embedding dimension (must be divisible by 8)
    ///
    /// # Panics
    ///
    /// Panics if `embedding_dim` is not divisible by 8.
    #[must_use]
    pub fn new(embedding_dim: usize) -> Self {
        assert!(
            embedding_dim.is_multiple_of(NUM_SUBVECTORS),
            "Embedding dimension {} must be divisible by {}",
            embedding_dim,
            NUM_SUBVECTORS
        );

        let codebook = Arc::new(Self::create_default_codebook(embedding_dim));
        Self { codebook }
    }

    /// Create a PQ8 encoder with a pre-trained codebook.
    pub fn with_codebook(codebook: Arc<PQ8Codebook>) -> Self {
        Self { codebook }
    }

    /// Create a default codebook with pseudo-random centroids.
    ///
    /// **WARNING: UNTRAINED CENTROIDS.** This codebook uses deterministic
    /// pseudo-random values from an LCG (Linear Congruential Generator) and
    /// does NOT reflect the actual distribution of any embedding model's output.
    /// Quantization recall will be significantly worse than a codebook trained
    /// via k-means on real embedding data. For production use, call
    /// `PQ8Encoder::with_codebook()` with a codebook trained on representative
    /// embeddings from the target model.
    ///
    /// Centroids are initialized using deterministic pseudo-random values
    /// covering the typical embedding range. This provides reasonable compression
    /// for general use, but should be replaced with a trained codebook for
    /// optimal recall on specific embedding distributions.
    ///
    /// # Algorithm
    ///
    /// Uses a simple linear congruential generator (LCG) for deterministic
    /// "random" values, ensuring reproducible behavior across runs.
    fn create_default_codebook(embedding_dim: usize) -> PQ8Codebook {
        let subvector_dim = embedding_dim / NUM_SUBVECTORS;
        let mut centroids = Vec::with_capacity(NUM_SUBVECTORS);

        // LCG parameters for pseudo-random generation (deterministic)
        let mut seed: u64 = 42;
        let lcg_next = |s: &mut u64| -> f32 {
            // LCG: x = (a * x + c) mod m
            *s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFFFFFF;
            // Map to [-1, 1] range
            (*s as f32 / 0x7FFFFFFF as f32) * 2.0 - 1.0
        };

        for _ in 0..NUM_SUBVECTORS {
            let mut subvector_centroids = Vec::with_capacity(NUM_CENTROIDS);
            for _ in 0..NUM_CENTROIDS {
                // Generate centroid with varied values per dimension
                let centroid: Vec<f32> = (0..subvector_dim).map(|_| lcg_next(&mut seed)).collect();
                subvector_centroids.push(centroid);
            }
            centroids.push(subvector_centroids);
        }

        PQ8Codebook {
            embedding_dim,
            num_subvectors: NUM_SUBVECTORS,
            num_centroids: NUM_CENTROIDS,
            centroids,
            codebook_id: 0, // Default codebook ID
        }
    }

    /// Quantize an f32 embedding vector to PQ-8 format.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The f32 embedding vector to compress
    ///
    /// # Returns
    ///
    /// `QuantizedEmbedding` with 8 bytes (32x compression).
    ///
    /// # Errors
    ///
    /// - `EmptyEmbedding` if input is empty
    /// - `ContainsNaN` if input has NaN values
    /// - `ContainsInfinity` if input has infinite values
    /// - `DimensionNotDivisible` if dimension not divisible by 8
    /// - `CodebookDimensionMismatch` if dimension doesn't match codebook
    pub fn quantize(&self, embedding: &[f32]) -> Result<QuantizedEmbedding, PQ8QuantizationError> {
        // Validate input
        if embedding.is_empty() {
            return Err(PQ8QuantizationError::EmptyEmbedding);
        }

        // Check for NaN and infinity
        for (i, &val) in embedding.iter().enumerate() {
            if val.is_nan() {
                return Err(PQ8QuantizationError::ContainsNaN { index: i });
            }
            if val.is_infinite() {
                return Err(PQ8QuantizationError::ContainsInfinity { index: i });
            }
        }

        // Validate dimension
        let dim = embedding.len();
        if !dim.is_multiple_of(NUM_SUBVECTORS) {
            return Err(PQ8QuantizationError::DimensionNotDivisible { dim });
        }

        if dim != self.codebook.embedding_dim {
            return Err(PQ8QuantizationError::CodebookDimensionMismatch {
                expected: self.codebook.embedding_dim,
                got: dim,
            });
        }

        let subvector_dim = dim / NUM_SUBVECTORS;

        debug!(
            target: "quantization::pq8",
            dim = dim,
            subvector_dim = subvector_dim,
            codebook_id = self.codebook.codebook_id,
            "Quantizing to PQ-8"
        );

        // Quantize each subvector
        let mut data = Vec::with_capacity(NUM_SUBVECTORS);
        for sv_idx in 0..NUM_SUBVECTORS {
            let start = sv_idx * subvector_dim;
            let end = start + subvector_dim;
            let subvector = &embedding[start..end];

            // Find nearest centroid
            let centroid_idx = self.find_nearest_centroid(sv_idx, subvector);
            data.push(centroid_idx);
        }

        Ok(QuantizedEmbedding {
            method: QuantizationMethod::PQ8,
            original_dim: dim,
            data,
            metadata: QuantizationMetadata::PQ8 {
                codebook_id: self.codebook.codebook_id,
                num_subvectors: NUM_SUBVECTORS as u8,
            },
        })
    }

    /// Dequantize a PQ-8 embedding back to f32 values.
    ///
    /// # Arguments
    ///
    /// * `quantized` - The quantized embedding to decompress
    ///
    /// # Returns
    ///
    /// Reconstructed f32 vector (approximately equal to original).
    ///
    /// # Errors
    ///
    /// - `InvalidMetadata` if metadata is not PQ8 type
    /// - `InvalidDataLength` if data is not 8 bytes
    pub fn dequantize(
        &self,
        quantized: &QuantizedEmbedding,
    ) -> Result<Vec<f32>, PQ8QuantizationError> {
        // Validate metadata
        match &quantized.metadata {
            QuantizationMetadata::PQ8 {
                codebook_id,
                num_subvectors,
            } => {
                if *codebook_id != self.codebook.codebook_id {
                    warn!(
                        target: "quantization::pq8",
                        expected_codebook = self.codebook.codebook_id,
                        got_codebook = codebook_id,
                        "Codebook ID mismatch - reconstruction may be inaccurate"
                    );
                }
                if *num_subvectors as usize != NUM_SUBVECTORS {
                    return Err(PQ8QuantizationError::InvalidMetadata {
                        expected: "PQ8 with 8 subvectors",
                        got: format!("PQ8 with {} subvectors", num_subvectors),
                    });
                }
            }
            other => {
                return Err(PQ8QuantizationError::InvalidMetadata {
                    expected: "PQ8",
                    got: format!("{:?}", other),
                });
            }
        }

        // Validate data length
        if quantized.data.len() != NUM_SUBVECTORS {
            return Err(PQ8QuantizationError::InvalidDataLength {
                expected: NUM_SUBVECTORS,
                got: quantized.data.len(),
            });
        }

        let subvector_dim = quantized.original_dim / NUM_SUBVECTORS;

        debug!(
            target: "quantization::pq8",
            dim = quantized.original_dim,
            subvector_dim = subvector_dim,
            "Dequantizing from PQ-8"
        );

        // Reconstruct embedding from centroid indices
        let mut result = Vec::with_capacity(quantized.original_dim);
        for (sv_idx, &centroid_idx) in quantized.data.iter().enumerate() {
            let centroid = &self.codebook.centroids[sv_idx][centroid_idx as usize];
            result.extend_from_slice(centroid);
        }

        Ok(result)
    }

    /// Find the nearest centroid index for a subvector.
    ///
    /// Uses squared Euclidean distance for efficiency.
    fn find_nearest_centroid(&self, subvector_idx: usize, subvector: &[f32]) -> u8 {
        let centroids = &self.codebook.centroids[subvector_idx];
        let mut min_dist = f32::MAX;
        let mut best_idx: u8 = 0;

        for (idx, centroid) in centroids.iter().enumerate() {
            let dist = self.squared_distance(subvector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_idx = idx as u8;
            }
        }

        best_idx
    }

    /// Compute squared Euclidean distance between two vectors.
    #[inline]
    fn squared_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }

    /// Get the codebook used by this encoder.
    pub fn codebook(&self) -> &PQ8Codebook {
        &self.codebook
    }

    /// Get the expected compression ratio.
    ///
    /// PQ-8 stores 8 centroid indices (1 byte each = 8 bytes total).
    /// For a D-dimensional f32 vector (D * 4 bytes), the actual ratio is D*4/8 = D/2.
    /// This constant returns 32.0 as a reference ratio assuming D=64 subvector elements,
    /// matching the QuantizationMethod::PQ8 constant used in the router.
    /// The true ratio is dimension-dependent: D=1024 -> 512x, D=768 -> 384x.
    #[must_use]
    pub const fn compression_ratio() -> f32 {
        32.0
    }
}

impl Default for PQ8Encoder {
    /// Create default encoder for 1024D embeddings (E1_Semantic dimension).
    fn default() -> Self {
        Self::new(1024)
    }
}
