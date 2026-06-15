//! Validation logic for model embeddings.
//!
//! This module provides validation methods to ensure embeddings
//! meet all requirements before use in the pipeline.

use crate::error::{EmbeddingError, EmbeddingResult};

use super::ModelEmbedding;

impl ModelEmbedding {
    /// Validates the embedding against model requirements.
    ///
    /// # Validation Rules
    /// 1. Vector dimension must match `model_id.dimension()` (or `projected_dimension()` if projected)
    /// 2. No NaN values allowed in vector
    /// 3. No Inf values allowed in vector
    /// 4. Vector must not be empty
    ///
    /// # Errors
    /// - `EmbeddingError::EmptyInput` if vector is empty
    /// - `EmbeddingError::InvalidDimension` if dimension doesn't match model
    /// - `EmbeddingError::InvalidValue` if NaN or Inf values found
    ///
    /// # Fail Fast
    /// This method fails immediately on first error - no partial validation.
    pub fn validate(&self) -> EmbeddingResult<()> {
        // Rule 1: Vector must not be empty
        if self.vector.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Rule 2: Check dimension matches expected
        let expected_dim = if self.is_projected {
            self.model_id.projected_dimension()
        } else {
            self.model_id.dimension()
        };

        if self.vector.len() != expected_dim {
            return Err(EmbeddingError::InvalidDimension {
                expected: expected_dim,
                actual: self.vector.len(),
            });
        }

        // Rule 3 & 4: Check for NaN and Inf values (fail fast)
        for (idx, &val) in self.vector.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(EmbeddingError::InvalidValue {
                    index: idx,
                    value: val,
                });
            }
        }

        Ok(())
    }

    /// Validates attention weights if present.
    ///
    /// # Arguments
    /// * `expected_token_count` - The number of input tokens
    ///
    /// # Errors
    /// - `EmbeddingError::InvalidDimension` if attention weights length != token count
    /// - `EmbeddingError::InvalidValue` if NaN/Inf in attention weights
    pub fn validate_attention(&self, expected_token_count: usize) -> EmbeddingResult<()> {
        if let Some(ref weights) = self.attention_weights {
            if weights.len() != expected_token_count {
                return Err(EmbeddingError::InvalidDimension {
                    expected: expected_token_count,
                    actual: weights.len(),
                });
            }

            for (idx, &val) in weights.iter().enumerate() {
                if val.is_nan() || val.is_infinite() {
                    return Err(EmbeddingError::InvalidValue {
                        index: idx,
                        value: val,
                    });
                }
            }
        }
        Ok(())
    }
}
