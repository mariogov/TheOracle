//! Quantization traits.

use super::types::{Precision, QuantizationError, QuantizedEmbedding};

/// Trait for types that can be quantized.
///
/// Implementors must provide quantization and dequantization methods
/// that preserve semantic ordering for similarity search.
pub trait Quantizable: Sized {
    /// Quantize the embedding to the specified precision.
    ///
    /// # Arguments
    ///
    /// * `precision` - Target quantization precision
    ///
    /// # Returns
    ///
    /// * `Ok(QuantizedEmbedding)` - Successfully quantized embedding
    /// * `Err(QuantizationError)` - Quantization failed
    ///
    /// # Invariants
    ///
    /// - Input MUST NOT contain NaN or Infinity values
    /// - Output MUST be deterministic (same input = same output)
    /// - Relative ordering of values SHOULD be preserved
    fn quantize(&self, precision: Precision) -> Result<QuantizedEmbedding, QuantizationError>;

    /// Dequantize back to the original type.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Previously quantized embedding
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - Reconstructed embedding
    /// * `Err(QuantizationError)` - Dequantization failed
    ///
    /// # Accuracy
    ///
    /// - Int8: RMSE < 1%
    /// - Int4: RMSE < 5%
    /// - Fp16: RMSE < 0.01%
    fn dequantize(quantized: &QuantizedEmbedding) -> Result<Self, QuantizationError>;

    /// Get the expected dimension for this embedding type.
    fn expected_dim(&self) -> usize;

    /// Validate the embedding contains no invalid values.
    fn validate(&self) -> Result<(), QuantizationError>;
}
