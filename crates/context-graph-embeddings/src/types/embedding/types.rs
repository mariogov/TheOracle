//! Core types for single model embedding output.
//!
//! This module provides the `ModelEmbedding` struct which represents
//! the output from a single embedding model in the 13-model pipeline.

use crate::types::ModelId;

/// Represents the embedding output from a single model.
///
/// # Fields
/// - `model_id`: Which of the 13 models produced this embedding
/// - `vector`: The embedding vector (f32 for GPU compatibility)
/// - `latency_us`: Time taken to generate embedding in microseconds
/// - `attention_weights`: Optional attention scores for interpretability
/// - `is_projected`: Whether vector has been projected to standard dimension
///
/// # Example
/// ```rust
/// use context_graph_embeddings::types::{ModelId, ModelEmbedding};
///
/// let embedding = ModelEmbedding::new(
///     ModelId::Semantic,
///     vec![0.1, 0.2, 0.3],  // simplified - real would be 1024 dims
///     1500,
/// );
/// assert_eq!(embedding.dimension(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ModelEmbedding {
    /// The model that produced this embedding (E1-E12)
    pub model_id: ModelId,

    /// The embedding vector - f32 for GPU compatibility
    pub vector: Vec<f32>,

    /// Generation latency in microseconds
    pub latency_us: u64,

    /// Optional attention weights for interpretability
    /// Length must match input token count when present
    pub attention_weights: Option<Vec<f32>>,

    /// Whether this vector has been projected to standard dimension
    pub is_projected: bool,
}

impl ModelEmbedding {
    /// Creates a new ModelEmbedding.
    ///
    /// # Arguments
    /// * `model_id` - The model that produced this embedding
    /// * `vector` - The raw embedding vector
    /// * `latency_us` - Generation time in microseconds
    ///
    /// # Note
    /// This does NOT validate the embedding. Call `validate()` after creation
    /// to ensure the embedding meets all requirements.
    #[inline]
    pub fn new(model_id: ModelId, vector: Vec<f32>, latency_us: u64) -> Self {
        Self {
            model_id,
            vector,
            latency_us,
            attention_weights: None,
            is_projected: false,
        }
    }

    /// Creates a new ModelEmbedding with attention weights.
    ///
    /// # Arguments
    /// * `model_id` - The model that produced this embedding
    /// * `vector` - The raw embedding vector
    /// * `latency_us` - Generation time in microseconds
    /// * `attention_weights` - Attention scores from the model
    #[inline]
    pub fn with_attention(
        model_id: ModelId,
        vector: Vec<f32>,
        latency_us: u64,
        attention_weights: Vec<f32>,
    ) -> Self {
        Self {
            model_id,
            vector,
            latency_us,
            attention_weights: Some(attention_weights),
            is_projected: false,
        }
    }

    /// Returns the dimension of the embedding vector.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Returns true if the embedding vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vector.is_empty()
    }

    /// Marks this embedding as projected to standard dimension.
    ///
    /// # Note
    /// This should be called after projection to update validation expectations.
    #[inline]
    pub fn set_projected(&mut self, projected: bool) {
        self.is_projected = projected;
    }
}

impl Default for ModelEmbedding {
    fn default() -> Self {
        Self {
            model_id: ModelId::Semantic,
            vector: Vec::new(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        }
    }
}
