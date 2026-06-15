//! Binary quantization tests (IMPLEMENTED).

use crate::quantization::router::QuantizationRouter;
use crate::quantization::types::QuantizationMethod;
use crate::types::ModelId;

#[test]
fn test_binary_quantization_e9_hdc() {
    let router = QuantizationRouter::new();

    // E9_HDC uses Binary quantization
    let embedding: Vec<f32> = (0..10000)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let quantized = router
        .quantize(ModelId::Hdc, &embedding)
        .expect("Binary quantization should succeed");

    assert_eq!(quantized.method, QuantizationMethod::Binary);
    assert_eq!(quantized.original_dim, 10000);
    // 10000 bits = 1250 bytes
    assert_eq!(quantized.data.len(), 1250);
}

#[test]
fn test_binary_round_trip() {
    let router = QuantizationRouter::new();

    // Create input with known pattern
    let input: Vec<f32> = (0..1024)
        .map(|i| if i % 3 == 0 { 0.5 } else { -0.5 })
        .collect();

    let quantized = router.quantize(ModelId::Hdc, &input).expect("quantize");

    let reconstructed = router
        .dequantize(ModelId::Hdc, &quantized)
        .expect("dequantize");

    // VERIFICATION: All signs must match (binary preserves sign only)
    for (i, (&orig, &recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
        let orig_positive = orig >= 0.0;
        let recon_positive = recon >= 0.0;
        assert_eq!(
            orig_positive, recon_positive,
            "Sign mismatch at index {}: orig={}, recon={}",
            i, orig, recon
        );
    }
}
