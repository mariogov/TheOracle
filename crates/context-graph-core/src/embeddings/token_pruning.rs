//! E12: Token-level embedding for late interaction (ColBERT-style).
//!
//! Per SPEC-E12-QUANT-001 and TECH-E12-QUANT-001.
//! TASK-L03: Implements Quantizable trait for TokenPruningEmbedding.

use crate::quantization::{Precision, Quantizable, QuantizationError, QuantizedEmbedding};

/// E12: Token-level embedding for late interaction (ColBERT-style).
///
/// Stores per-token embeddings of dimension 128.
/// Used for MaxSim scoring: max_j(cos(q_i, d_j)) for each query token.
#[derive(Debug, Clone)]
pub struct TokenPruningEmbedding {
    /// Per-token embeddings, shape: [num_tokens, 128]
    /// Stored as flattened Vec for efficiency.
    pub values: Vec<f32>,

    /// Number of tokens.
    pub num_tokens: usize,

    /// Per-token dimension (always 128 for E12).
    pub per_token_dim: usize,
}

impl TokenPruningEmbedding {
    /// E12 per-token dimension per constitution.
    pub const PER_TOKEN_DIM: usize = 128;

    /// Create a new token pruning embedding.
    ///
    /// # Arguments
    ///
    /// * `values` - Flattened token embeddings
    /// * `num_tokens` - Number of tokens
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != num_tokens * 128`.
    #[track_caller]
    pub fn new(values: Vec<f32>, num_tokens: usize) -> Self {
        assert_eq!(
            values.len(),
            num_tokens * Self::PER_TOKEN_DIM,
            "TokenPruningEmbedding: values.len()={} must equal num_tokens={} * {}",
            values.len(),
            num_tokens,
            Self::PER_TOKEN_DIM
        );
        Self {
            values,
            num_tokens,
            per_token_dim: Self::PER_TOKEN_DIM,
        }
    }

    /// Create from 2D shape (for testing).
    pub fn from_2d(tokens: Vec<Vec<f32>>) -> Self {
        let num_tokens = tokens.len();
        let values: Vec<f32> = tokens.into_iter().flatten().collect();
        Self::new(values, num_tokens)
    }

    /// Get embedding for a specific token.
    pub fn token_embedding(&self, token_idx: usize) -> Option<&[f32]> {
        if token_idx >= self.num_tokens {
            return None;
        }
        let start = token_idx * self.per_token_dim;
        let end = start + self.per_token_dim;
        Some(&self.values[start..end])
    }

    /// Get token importance scores (L2 norms).
    pub fn token_importance(&self) -> Vec<f32> {
        (0..self.num_tokens)
            .map(|i| {
                let emb = self.token_embedding(i).unwrap();
                emb.iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect()
    }

    /// Get top-k tokens by importance.
    pub fn top_k_tokens(&self, k: usize) -> Vec<usize> {
        let mut importance: Vec<(usize, f32)> =
            self.token_importance().into_iter().enumerate().collect();
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        importance.into_iter().take(k).map(|(i, _)| i).collect()
    }

    // ========================================================================
    // Quantization Helper Methods
    // ========================================================================

    /// Quantize to INT8.
    fn quantize_int8(&self, min_val: f32, max_val: f32) -> (f32, i32, Vec<u8>) {
        let range = max_val - min_val;
        let scale = range / 255.0;
        let zero_point = (-128.0 - min_val / scale).round() as i32;

        let data: Vec<u8> = self
            .values
            .iter()
            .map(|&v| {
                let quantized = ((v / scale) + zero_point as f32)
                    .round()
                    .clamp(-128.0, 127.0) as i8;
                quantized as u8
            })
            .collect();

        (scale, zero_point, data)
    }

    /// Quantize to INT4 (packed nibbles).
    fn quantize_int4(&self, min_val: f32, max_val: f32) -> (f32, i32, Vec<u8>) {
        let range = max_val - min_val;
        let scale = range / 15.0;
        let zero_point = (-8.0 - min_val / scale).round() as i32;

        // Pack two 4-bit values per byte
        let mut data = Vec::with_capacity(self.values.len().div_ceil(2));
        for chunk in self.values.chunks(2) {
            let v0 = ((chunk[0] / scale) + zero_point as f32)
                .round()
                .clamp(-8.0, 7.0) as i8;
            let v1 = if chunk.len() > 1 {
                ((chunk[1] / scale) + zero_point as f32)
                    .round()
                    .clamp(-8.0, 7.0) as i8
            } else {
                0i8
            };
            // Pack: high nibble = v0, low nibble = v1
            let packed = ((v0 as u8 & 0x0F) << 4) | (v1 as u8 & 0x0F);
            data.push(packed);
        }

        (scale, zero_point, data)
    }

    /// Quantize to FP16.
    fn quantize_fp16(&self) -> (f32, i32, Vec<u8>) {
        let data: Vec<u8> = self
            .values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();

        (1.0, 0, data)
    }

    /// Handle constant embedding (all same value).
    fn quantize_constant(
        &self,
        precision: Precision,
        value: f32,
    ) -> Result<QuantizedEmbedding, QuantizationError> {
        let data = match precision {
            Precision::Int8 => vec![0u8; self.values.len()],
            Precision::Int4 => vec![0u8; self.values.len().div_ceil(2)],
            Precision::Fp16 => {
                let half_val = half::f16::from_f32(value);
                half_val
                    .to_le_bytes()
                    .iter()
                    .cycle()
                    .take(self.values.len() * 2)
                    .copied()
                    .collect()
            }
            _ => unreachable!(),
        };

        Ok(QuantizedEmbedding {
            data,
            scale: 1.0,
            zero_point: 0,
            precision,
            original_dim: self.values.len(),
            embedding_type: "token_pruning".to_string(),
            token_count: Some(self.num_tokens),
        })
    }

    // ========================================================================
    // Dequantization Helper Methods
    // ========================================================================

    /// Dequantize from INT8.
    fn dequantize_int8(quantized: &QuantizedEmbedding) -> Vec<f32> {
        quantized
            .data
            .iter()
            .map(|&byte| {
                let quantized_val = byte as i8;
                (quantized_val as f32 - quantized.zero_point as f32) * quantized.scale
            })
            .collect()
    }

    /// Dequantize from INT4.
    fn dequantize_int4(quantized: &QuantizedEmbedding) -> Vec<f32> {
        let mut values = Vec::with_capacity(quantized.original_dim);
        for &byte in &quantized.data {
            // Unpack high nibble (sign-extend)
            let v0 = ((byte >> 4) as i8) << 4 >> 4;
            values.push((v0 as f32 - quantized.zero_point as f32) * quantized.scale);

            if values.len() < quantized.original_dim {
                // Unpack low nibble (sign-extend)
                let v1 = (byte as i8) << 4 >> 4;
                values.push((v1 as f32 - quantized.zero_point as f32) * quantized.scale);
            }
        }
        values.truncate(quantized.original_dim);
        values
    }

    /// Dequantize from FP16.
    fn dequantize_fp16(quantized: &QuantizedEmbedding) -> Vec<f32> {
        quantized
            .data
            .chunks(2)
            .map(|chunk| {
                let bytes = [chunk[0], chunk.get(1).copied().unwrap_or(0)];
                half::f16::from_le_bytes(bytes).to_f32()
            })
            .collect()
    }
}

impl Quantizable for TokenPruningEmbedding {
    fn quantize(&self, precision: Precision) -> Result<QuantizedEmbedding, QuantizationError> {
        // 1. Validate input
        self.validate()?;

        // 2. Check precision is supported
        if !precision.is_quantizable() {
            return Err(QuantizationError::UnsupportedPrecision { precision });
        }

        // 3. Compute min/max for scale calculation
        let (min_val, max_val) = self
            .values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

        // 4. Handle edge case: all values are the same
        let range = max_val - min_val;
        if range < f32::EPSILON {
            return self.quantize_constant(precision, min_val);
        }

        // 5. Quantize based on precision
        let (scale, zero_point, data) = match precision {
            Precision::Int8 => self.quantize_int8(min_val, max_val),
            Precision::Int4 => self.quantize_int4(min_val, max_val),
            Precision::Fp16 => self.quantize_fp16(),
            Precision::Fp32 => unreachable!(), // Checked above
        };

        Ok(QuantizedEmbedding {
            data,
            scale,
            zero_point,
            precision,
            original_dim: self.values.len(),
            embedding_type: "token_pruning".to_string(),
            token_count: Some(self.num_tokens),
        })
    }

    fn dequantize(quantized: &QuantizedEmbedding) -> Result<Self, QuantizationError> {
        // 1. Validate embedding type
        if quantized.embedding_type != "token_pruning" {
            return Err(QuantizationError::InvalidData {
                reason: format!(
                    "Expected 'token_pruning', got '{}'",
                    quantized.embedding_type
                ),
            });
        }

        // 2. Get token count
        let num_tokens = quantized
            .token_count
            .ok_or_else(|| QuantizationError::InvalidData {
                reason: "token_count required for TokenPruningEmbedding".to_string(),
            })?;

        // 3. Dequantize based on precision
        let values: Vec<f32> = match quantized.precision {
            Precision::Int8 => Self::dequantize_int8(quantized),
            Precision::Int4 => Self::dequantize_int4(quantized),
            Precision::Fp16 => Self::dequantize_fp16(quantized),
            _ => {
                return Err(QuantizationError::UnsupportedPrecision {
                    precision: quantized.precision,
                })
            }
        };

        // 4. Validate dimension
        if values.len() != quantized.original_dim {
            return Err(QuantizationError::DimensionMismatch {
                expected: quantized.original_dim,
                actual: values.len(),
            });
        }

        Ok(Self::new(values, num_tokens))
    }

    fn expected_dim(&self) -> usize {
        self.num_tokens * self.per_token_dim
    }

    fn validate(&self) -> Result<(), QuantizationError> {
        for (i, &v) in self.values.iter().enumerate() {
            if v.is_nan() {
                return Err(QuantizationError::NaN { index: i });
            }
            if v.is_infinite() {
                let sign = if v > 0.0 { "+" } else { "-" };
                return Err(QuantizationError::Infinity { index: i, sign });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_embedding(num_tokens: usize) -> TokenPruningEmbedding {
        let values: Vec<f32> = (0..(num_tokens * 128))
            .map(|i| (i as f32 / 1000.0) - 0.5)
            .collect();
        TokenPruningEmbedding::new(values, num_tokens)
    }

    #[test]
    fn test_new_valid() {
        let emb = test_embedding(10);
        assert_eq!(emb.num_tokens, 10);
        assert_eq!(emb.values.len(), 1280);
    }

    #[test]
    #[should_panic]
    fn test_new_invalid_size() {
        TokenPruningEmbedding::new(vec![0.0; 100], 10); // Should be 1280
    }

    #[test]
    fn test_from_2d() {
        let tokens: Vec<Vec<f32>> = (0..5).map(|_| vec![0.1; 128]).collect();
        let emb = TokenPruningEmbedding::from_2d(tokens);
        assert_eq!(emb.num_tokens, 5);
        assert_eq!(emb.values.len(), 640);
    }

    #[test]
    fn test_token_embedding() {
        let emb = test_embedding(10);
        let token_0 = emb.token_embedding(0).unwrap();
        assert_eq!(token_0.len(), 128);

        let token_9 = emb.token_embedding(9).unwrap();
        assert_eq!(token_9.len(), 128);

        assert!(emb.token_embedding(10).is_none());
    }

    #[test]
    fn test_quantize_int8() {
        let emb = test_embedding(10);
        let quantized = emb.quantize(Precision::Int8).unwrap();

        assert_eq!(quantized.precision, Precision::Int8);
        assert_eq!(quantized.data.len(), 1280);
        assert!((quantized.compression_ratio() - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_int4() {
        let emb = test_embedding(10);
        let quantized = emb.quantize(Precision::Int4).unwrap();

        assert_eq!(quantized.precision, Precision::Int4);
        assert_eq!(quantized.data.len(), 640); // Packed nibbles
        assert!((quantized.compression_ratio() - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_fp16() {
        let emb = test_embedding(10);
        let quantized = emb.quantize(Precision::Fp16).unwrap();

        assert_eq!(quantized.precision, Precision::Fp16);
        assert_eq!(quantized.data.len(), 2560); // 2 bytes per value
        assert!((quantized.compression_ratio() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_fp32_unsupported() {
        let emb = test_embedding(10);
        let result = emb.quantize(Precision::Fp32);
        assert!(matches!(
            result,
            Err(QuantizationError::UnsupportedPrecision { .. })
        ));
    }

    #[test]
    fn test_roundtrip_int8() {
        let emb = test_embedding(10);
        let quantized = emb.quantize(Precision::Int8).unwrap();
        let reconstructed = TokenPruningEmbedding::dequantize(&quantized).unwrap();

        assert_eq!(reconstructed.num_tokens, emb.num_tokens);
        assert_eq!(reconstructed.values.len(), emb.values.len());

        // Check RMSE < 1%
        let nrmse = compute_nrmse(&emb.values, &reconstructed.values);
        assert!(nrmse < 0.01, "INT8 NRMSE {} exceeds 1%", nrmse);
    }

    #[test]
    fn test_roundtrip_int4() {
        let emb = test_embedding(10);
        let quantized = emb.quantize(Precision::Int4).unwrap();
        let reconstructed = TokenPruningEmbedding::dequantize(&quantized).unwrap();

        let nrmse = compute_nrmse(&emb.values, &reconstructed.values);
        assert!(nrmse < 0.05, "INT4 NRMSE {} exceeds 5%", nrmse);
    }

    #[test]
    fn test_roundtrip_fp16() {
        let emb = test_embedding(10);
        let quantized = emb.quantize(Precision::Fp16).unwrap();
        let reconstructed = TokenPruningEmbedding::dequantize(&quantized).unwrap();

        let nrmse = compute_nrmse(&emb.values, &reconstructed.values);
        assert!(nrmse < 0.001, "FP16 NRMSE {} exceeds 0.1%", nrmse);
    }

    #[test]
    fn test_validate_nan() {
        let mut emb = test_embedding(10);
        emb.values[500] = f32::NAN;

        let result = emb.validate();
        assert!(matches!(result, Err(QuantizationError::NaN { index: 500 })));
    }

    #[test]
    fn test_validate_infinity() {
        let mut emb = test_embedding(10);
        emb.values[100] = f32::INFINITY;

        let result = emb.validate();
        assert!(matches!(
            result,
            Err(QuantizationError::Infinity {
                index: 100,
                sign: "+"
            })
        ));
    }

    #[test]
    fn test_token_importance() {
        let emb = test_embedding(10);
        let importance = emb.token_importance();

        assert_eq!(importance.len(), 10);
        for &imp in &importance {
            assert!(imp >= 0.0);
            assert!(!imp.is_nan());
        }
    }

    #[test]
    fn test_top_k_tokens() {
        let emb = test_embedding(10);
        let top_k = emb.top_k_tokens(3);

        assert_eq!(top_k.len(), 3);
        for &idx in &top_k {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_constant_embedding() {
        let values = vec![0.5f32; 1280];
        let emb = TokenPruningEmbedding::new(values, 10);

        let quantized = emb.quantize(Precision::Int8).unwrap();
        assert!(quantized.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_dequantize_wrong_type() {
        let emb = test_embedding(10);
        let mut quantized = emb.quantize(Precision::Int8).unwrap();
        quantized.embedding_type = "wrong_type".to_string();

        let result = TokenPruningEmbedding::dequantize(&quantized);
        assert!(matches!(result, Err(QuantizationError::InvalidData { .. })));
    }

    #[test]
    fn test_dequantize_missing_token_count() {
        let emb = test_embedding(10);
        let mut quantized = emb.quantize(Precision::Int8).unwrap();
        quantized.token_count = None;

        let result = TokenPruningEmbedding::dequantize(&quantized);
        assert!(matches!(result, Err(QuantizationError::InvalidData { .. })));
    }

    // Helper function for NRMSE calculation in tests
    fn compute_nrmse(original: &[f32], reconstructed: &[f32]) -> f32 {
        if original.len() != reconstructed.len() || original.is_empty() {
            return f32::NAN;
        }

        let sum_sq: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let rmse = (sum_sq / original.len() as f32).sqrt();

        let (min, max) = original
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

        let range = max - min;
        if range < f32::EPSILON {
            return 0.0;
        }

        rmse / range
    }
}
