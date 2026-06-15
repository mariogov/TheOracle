//! Mathematical operations for model embeddings.
//!
//! This module provides normalization, similarity computation,
//! and other vector operations for embeddings.

use crate::error::{EmbeddingError, EmbeddingResult};

use super::ModelEmbedding;

impl ModelEmbedding {
    /// Consumes the embedding and returns the underlying vector.
    ///
    /// # Returns
    /// The owned embedding vector.
    ///
    /// # Example
    /// ```rust
    /// use context_graph_embeddings::types::{ModelId, ModelEmbedding};
    ///
    /// let embedding = ModelEmbedding::new(ModelId::Semantic, vec![0.1, 0.2, 0.3], 1000);
    /// let vector = embedding.into_vec();
    /// assert_eq!(vector.len(), 3);
    /// ```
    #[inline]
    pub fn into_vec(self) -> Vec<f32> {
        self.vector
    }

    /// Calculates the L2 (Euclidean) norm of the vector.
    ///
    /// # Returns
    /// The square root of the sum of squared elements.
    /// Returns 0.0 for empty vectors.
    ///
    /// # Performance
    /// Uses SIMD-friendly loop pattern for GPU/CPU optimization.
    #[inline]
    pub fn l2_norm(&self) -> f32 {
        if self.vector.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = self.vector.iter().map(|x| x * x).sum();

        sum_squares.sqrt()
    }

    /// Normalizes the vector to unit length (L2 norm = 1.0).
    ///
    /// # Behavior
    /// - After normalization, `l2_norm()` returns approximately 1.0
    /// - Zero vectors remain unchanged (avoids division by zero)
    /// - Empty vectors remain unchanged
    ///
    /// # Panics
    /// Does not panic. Zero vectors are handled gracefully.
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();

        // Avoid division by zero - zero vectors stay zero
        if norm > f32::EPSILON {
            for val in &mut self.vector {
                *val /= norm;
            }
        }
    }

    /// Returns a normalized copy of this embedding.
    ///
    /// # Returns
    /// A new ModelEmbedding with the same metadata but normalized vector.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Checks if the vector is normalized (L2 norm approximately 1.0).
    ///
    /// # Arguments
    /// * `epsilon` - Tolerance for floating point comparison (default: 1e-6)
    #[inline]
    pub fn is_normalized(&self, epsilon: f32) -> bool {
        if self.vector.is_empty() {
            return false;
        }
        (self.l2_norm() - 1.0).abs() < epsilon
    }

    /// Computes cosine similarity with another embedding.
    ///
    /// # Arguments
    /// * `other` - The embedding to compare against
    ///
    /// # Returns
    /// Cosine similarity in range [-1.0, 1.0]
    ///
    /// # Errors
    /// - `EmbeddingError::InvalidDimension` if dimensions don't match
    /// - `EmbeddingError::EmptyInput` if either vector is empty
    pub fn cosine_similarity(&self, other: &Self) -> EmbeddingResult<f32> {
        if self.vector.is_empty() || other.vector.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        if self.vector.len() != other.vector.len() {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.vector.len(),
                actual: other.vector.len(),
            });
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();

        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }
}
