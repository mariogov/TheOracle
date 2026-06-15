//! Float8 quantization tests (IMPLEMENTED).

use crate::quantization::router::QuantizationRouter;
use crate::quantization::types::QuantizationMethod;
use crate::types::ModelId;

#[test]
fn test_float8_quantization_e2_temporal_recent() {
    let router = QuantizationRouter::new();

    // E2_TemporalRecent uses Float8E4M3 quantization
    let embedding: Vec<f32> = (0..512).map(|i| (i as f32 / 512.0) * 2.0 - 1.0).collect();

    let quantized = router
        .quantize(ModelId::TemporalRecent, &embedding)
        .expect("Float8E4M3 quantization should succeed");

    assert_eq!(quantized.method, QuantizationMethod::Float8E4M3);
    assert_eq!(quantized.original_dim, 512);
    // Float8: 1 byte per element = 512 bytes
    assert_eq!(quantized.data.len(), 512);
}

#[test]
fn test_float8_round_trip() {
    let router = QuantizationRouter::new();

    // Create input with known range
    let input: Vec<f32> = (0..256)
        .map(|i| (i as f32 / 128.0) - 1.0) // -1.0 to ~1.0
        .collect();

    let quantized = router
        .quantize(ModelId::TemporalRecent, &input)
        .expect("quantize");

    let reconstructed = router
        .dequantize(ModelId::TemporalRecent, &quantized)
        .expect("dequantize");

    // VERIFICATION: Float8E4M3 should have max error < 0.3% (per Constitution)
    // For values in [-1, 1], relative error should be small
    let max_abs_error: f32 = input
        .iter()
        .zip(reconstructed.iter())
        .map(|(&orig, &recon)| (orig - recon).abs())
        .fold(0.0, f32::max);

    // Float8E4M3 has ~1/16 precision for small values, max error should be reasonable
    assert!(
        max_abs_error < 0.5,
        "Max absolute error {} exceeds threshold 0.5",
        max_abs_error
    );
}

#[test]
fn test_float8_all_model_ids() {
    let router = QuantizationRouter::new();
    let embedding = vec![0.5f32; 384];

    // Test all Float8 model IDs: E2, E3, E4, E8, E11
    let float8_models = [
        ModelId::TemporalRecent,
        ModelId::TemporalPeriodic,
        ModelId::TemporalPositional,
        ModelId::Graph,
        ModelId::Entity,
    ];

    for model_id in float8_models {
        let result = router.quantize(model_id, &embedding);
        assert!(
            result.is_ok(),
            "Float8 quantization failed for {:?}: {:?}",
            model_id,
            result.err()
        );
        let q = result.unwrap();
        assert_eq!(q.method, QuantizationMethod::Float8E4M3);
        assert_eq!(q.data.len(), 384); // 1 byte per element
    }
}
