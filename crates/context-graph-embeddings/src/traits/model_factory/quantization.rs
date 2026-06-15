//! Quantization modes for memory reduction.
//!
//! Lower precision reduces memory and may increase throughput,
//! but can affect embedding quality.

use serde::{Deserialize, Serialize};

/// Quantization modes for memory reduction.
///
/// Lower precision reduces memory and may increase throughput,
/// but can affect embedding quality.
///
/// # RTX 5090 Tensor Core Support
///
/// | Mode | Memory | Speed | Quality | Tensor Core |
/// |------|--------|-------|---------|-------------|
/// | None | 100% | Baseline | 100% | FP32/TF32 |
/// | FP16 | 50% | ~2x | ~99.5% | Yes |
/// | BF16 | 50% | ~2x | ~99.5% | Yes |
/// | Int8 | 25% | ~3x | ~99% | Yes |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationMode {
    /// No quantization (FP32 weights).
    #[default]
    None,

    /// 8-bit integer quantization.
    /// Best memory savings, slight quality reduction.
    Int8,

    /// 16-bit floating point.
    /// Good balance of memory and quality.
    Fp16,

    /// Brain floating point 16-bit.
    /// Better for training, good for inference.
    Bf16,
}

impl QuantizationMode {
    /// Returns the memory multiplier relative to FP32.
    /// Example: FP16 returns 0.5 (50% of FP32 memory).
    pub fn memory_multiplier(&self) -> f32 {
        match self {
            QuantizationMode::None => 1.0,
            QuantizationMode::Int8 => 0.25,
            QuantizationMode::Fp16 | QuantizationMode::Bf16 => 0.5,
        }
    }

    /// Returns bytes per parameter.
    pub fn bytes_per_param(&self) -> usize {
        match self {
            QuantizationMode::None => 4,
            QuantizationMode::Fp16 | QuantizationMode::Bf16 => 2,
            QuantizationMode::Int8 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_default_is_none() {
        let quant = QuantizationMode::default();
        assert_eq!(quant, QuantizationMode::None);
    }

    #[test]
    fn test_quantization_memory_multiplier() {
        assert_eq!(QuantizationMode::None.memory_multiplier(), 1.0);
        assert_eq!(QuantizationMode::Int8.memory_multiplier(), 0.25);
        assert_eq!(QuantizationMode::Fp16.memory_multiplier(), 0.5);
        assert_eq!(QuantizationMode::Bf16.memory_multiplier(), 0.5);
    }

    #[test]
    fn test_quantization_bytes_per_param() {
        assert_eq!(QuantizationMode::None.bytes_per_param(), 4);
        assert_eq!(QuantizationMode::Fp16.bytes_per_param(), 2);
        assert_eq!(QuantizationMode::Bf16.bytes_per_param(), 2);
        assert_eq!(QuantizationMode::Int8.bytes_per_param(), 1);
    }

    #[test]
    fn test_quantization_serde_roundtrip() {
        let modes = [
            QuantizationMode::None,
            QuantizationMode::Int8,
            QuantizationMode::Fp16,
            QuantizationMode::Bf16,
        ];

        for mode in modes {
            let json = serde_json::to_string(&mode).unwrap();
            let restored: QuantizationMode = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, mode);
        }
    }

    #[test]
    fn test_quantization_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<QuantizationMode>();
    }
}
