//! Quantization types for embedding compression.
//!
//! Per SPEC-E12-QUANT-001 and TECH-E12-QUANT-001.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Quantization precision levels.
///
/// Each precision level offers different trade-offs between
/// compression ratio and reconstruction accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Precision {
    /// 4-bit integer quantization.
    /// Range: [-8, 7]
    /// Compression: ~8x
    /// Typical RMSE: < 5%
    Int4 = 4,

    /// 8-bit integer quantization.
    /// Range: [-128, 127]
    /// Compression: ~4x
    /// Typical RMSE: < 1%
    Int8 = 8,

    /// 16-bit floating point (IEEE 754 half precision).
    /// Range: IEEE 754 binary16
    /// Compression: ~2x
    /// Typical RMSE: < 0.01%
    Fp16 = 16,

    /// 32-bit floating point (no quantization, for reference).
    /// Range: IEEE 754 binary32
    /// Compression: 1x
    /// RMSE: 0%
    Fp32 = 32,
}

impl Precision {
    /// Get the number of bits for this precision.
    #[inline]
    pub const fn bits(self) -> u8 {
        self as u8
    }

    /// Get the compression ratio compared to FP32.
    pub fn compression_ratio(self) -> f32 {
        32.0 / self.bits() as f32
    }

    /// Get the value range for integer precisions.
    pub fn range(self) -> Option<(i32, i32)> {
        match self {
            Self::Int4 => Some((-8, 7)),
            Self::Int8 => Some((-128, 127)),
            _ => None,
        }
    }

    /// Get the expected RMSE threshold for this precision.
    pub fn rmse_threshold(self) -> f32 {
        match self {
            Self::Int4 => 0.05,   // 5%
            Self::Int8 => 0.01,   // 1%
            Self::Fp16 => 0.0001, // 0.01%
            Self::Fp32 => 0.0,    // 0%
        }
    }

    /// Check if this precision is supported for quantization.
    pub fn is_quantizable(self) -> bool {
        !matches!(self, Self::Fp32)
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int4 => write!(f, "INT4"),
            Self::Int8 => write!(f, "INT8"),
            Self::Fp16 => write!(f, "FP16"),
            Self::Fp32 => write!(f, "FP32"),
        }
    }
}

/// A quantized embedding with metadata for dequantization.
///
/// Stores the quantized data along with scale and zero-point
/// for linear dequantization: `original = (quantized - zero_point) * scale`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEmbedding {
    /// Quantized data bytes.
    /// - Int4: packed nibbles (2 values per byte)
    /// - Int8: one value per byte
    /// - Fp16: two bytes per value (little-endian)
    pub data: Vec<u8>,

    /// Scale factor for dequantization.
    /// For FP16, this is 1.0.
    pub scale: f32,

    /// Zero point for dequantization.
    /// For symmetric quantization, this is 0.
    pub zero_point: i32,

    /// Precision used for quantization.
    pub precision: Precision,

    /// Original embedding dimension.
    pub original_dim: usize,

    /// Original embedding type (for E12, this is "token_pruning").
    pub embedding_type: String,

    /// Number of tokens (for token-level embeddings like E12).
    /// None for fixed-dimension embeddings.
    pub token_count: Option<usize>,
}

impl QuantizedEmbedding {
    /// Get the compressed size in bytes.
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Get the original uncompressed size in bytes.
    pub fn uncompressed_size(&self) -> usize {
        self.original_dim * 4 // f32 = 4 bytes
    }

    /// Get the actual compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        if self.compressed_size() == 0 {
            return 0.0;
        }
        self.uncompressed_size() as f32 / self.compressed_size() as f32
    }

    /// Validate the quantized embedding structure.
    pub fn validate(&self) -> Result<(), QuantizationError> {
        // Check data size matches precision and dimension
        let expected_size = match self.precision {
            Precision::Int4 => self.original_dim.div_ceil(2),
            Precision::Int8 => self.original_dim,
            Precision::Fp16 => self.original_dim * 2,
            Precision::Fp32 => {
                return Err(QuantizationError::UnsupportedPrecision {
                    precision: self.precision,
                })
            }
        };

        if self.data.len() != expected_size {
            return Err(QuantizationError::InvalidData {
                reason: format!(
                    "Expected {} bytes for {} dim at {:?}, got {}",
                    expected_size,
                    self.original_dim,
                    self.precision,
                    self.data.len()
                ),
            });
        }

        Ok(())
    }
}

/// Errors during quantization/dequantization.
#[derive(Debug, Error, Clone)]
pub enum QuantizationError {
    /// Input contains NaN value.
    #[error("E_E12_QUANT_001: NaN value at index {index}")]
    NaN { index: usize },

    /// Input contains Infinity value.
    #[error("E_E12_QUANT_002: {sign}Infinity at index {index}")]
    Infinity { index: usize, sign: &'static str },

    /// Dimension mismatch.
    #[error("E_E12_QUANT_003: Expected dimension {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Unsupported precision.
    #[error("E_E12_QUANT_004: Unsupported precision {precision}. Supported: Int4, Int8, Fp16")]
    UnsupportedPrecision { precision: Precision },

    /// Quantization overflow (values clamped).
    #[error("E_E12_QUANT_005: {clamped_count} values clamped during quantization")]
    Overflow { clamped_count: usize },

    /// Invalid quantized data.
    #[error("E_E12_QUANT_006: Invalid quantized data: {reason}")]
    InvalidData { reason: String },
}

impl QuantizationError {
    /// Get the error code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NaN { .. } => "E_E12_QUANT_001",
            Self::Infinity { .. } => "E_E12_QUANT_002",
            Self::DimensionMismatch { .. } => "E_E12_QUANT_003",
            Self::UnsupportedPrecision { .. } => "E_E12_QUANT_004",
            Self::Overflow { .. } => "E_E12_QUANT_005",
            Self::InvalidData { .. } => "E_E12_QUANT_006",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_bits() {
        assert_eq!(Precision::Int4.bits(), 4);
        assert_eq!(Precision::Int8.bits(), 8);
        assert_eq!(Precision::Fp16.bits(), 16);
        assert_eq!(Precision::Fp32.bits(), 32);
    }

    #[test]
    fn test_precision_compression_ratio() {
        assert!((Precision::Int4.compression_ratio() - 8.0).abs() < 0.001);
        assert!((Precision::Int8.compression_ratio() - 4.0).abs() < 0.001);
        assert!((Precision::Fp16.compression_ratio() - 2.0).abs() < 0.001);
        assert!((Precision::Fp32.compression_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_precision_range() {
        assert_eq!(Precision::Int4.range(), Some((-8, 7)));
        assert_eq!(Precision::Int8.range(), Some((-128, 127)));
        assert_eq!(Precision::Fp16.range(), None);
        assert_eq!(Precision::Fp32.range(), None);
    }

    #[test]
    fn test_precision_rmse_threshold() {
        assert!((Precision::Int4.rmse_threshold() - 0.05).abs() < 0.001);
        assert!((Precision::Int8.rmse_threshold() - 0.01).abs() < 0.001);
        assert!((Precision::Fp16.rmse_threshold() - 0.0001).abs() < 0.00001);
    }

    #[test]
    fn test_precision_is_quantizable() {
        assert!(Precision::Int4.is_quantizable());
        assert!(Precision::Int8.is_quantizable());
        assert!(Precision::Fp16.is_quantizable());
        assert!(!Precision::Fp32.is_quantizable());
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", Precision::Int4), "INT4");
        assert_eq!(format!("{}", Precision::Int8), "INT8");
        assert_eq!(format!("{}", Precision::Fp16), "FP16");
        assert_eq!(format!("{}", Precision::Fp32), "FP32");
    }

    #[test]
    fn test_precision_serde() {
        let precision = Precision::Int8;
        let json = serde_json::to_string(&precision).unwrap();
        let parsed: Precision = serde_json::from_str(&json).unwrap();
        assert_eq!(precision, parsed);
    }

    #[test]
    fn test_quantized_embedding_compression_ratio() {
        let qe = QuantizedEmbedding {
            data: vec![0u8; 128],
            scale: 0.01,
            zero_point: 0,
            precision: Precision::Int8,
            original_dim: 128,
            embedding_type: "test".to_string(),
            token_count: None,
        };

        // 128 * 4 bytes (f32) / 128 bytes = 4.0
        assert!((qe.compression_ratio() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_quantized_embedding_validate_int8() {
        let qe = QuantizedEmbedding {
            data: vec![0u8; 128],
            scale: 0.01,
            zero_point: 0,
            precision: Precision::Int8,
            original_dim: 128,
            embedding_type: "test".to_string(),
            token_count: None,
        };
        assert!(qe.validate().is_ok());
    }

    #[test]
    fn test_quantized_embedding_validate_wrong_size() {
        let qe = QuantizedEmbedding {
            data: vec![0u8; 64], // Wrong size
            scale: 0.01,
            zero_point: 0,
            precision: Precision::Int8,
            original_dim: 128,
            embedding_type: "test".to_string(),
            token_count: None,
        };
        assert!(matches!(
            qe.validate(),
            Err(QuantizationError::InvalidData { .. })
        ));
    }

    #[test]
    fn test_quantization_error_codes() {
        assert_eq!(
            QuantizationError::NaN { index: 0 }.code(),
            "E_E12_QUANT_001"
        );
        assert_eq!(
            QuantizationError::Infinity {
                index: 0,
                sign: "+"
            }
            .code(),
            "E_E12_QUANT_002"
        );
        assert_eq!(
            QuantizationError::DimensionMismatch {
                expected: 128,
                actual: 64
            }
            .code(),
            "E_E12_QUANT_003"
        );
        assert_eq!(
            QuantizationError::UnsupportedPrecision {
                precision: Precision::Fp32
            }
            .code(),
            "E_E12_QUANT_004"
        );
    }

    #[test]
    fn test_quantization_error_display() {
        let err = QuantizationError::NaN { index: 42 };
        let msg = format!("{}", err);
        assert!(msg.contains("E_E12_QUANT_001"));
        assert!(msg.contains("42"));
    }
}
