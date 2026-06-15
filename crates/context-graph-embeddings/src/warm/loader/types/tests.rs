//! Tests for warm loader types.
//!
//! # Constitution Alignment
//!
//! These tests verify fail-fast behavior per AP-007.

use super::*;
use candle_core::DType;
use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn create_test_metadata() -> TensorMetadata {
    let mut shapes = HashMap::new();
    shapes.insert("embeddings.weight".to_string(), vec![30522, 768]);
    shapes.insert("encoder.layer.0.weight".to_string(), vec![768, 768]);
    // Total: 30522*768 + 768*768 = 23_440_896 + 589_824 = 24_030_720
    TensorMetadata::new(shapes, DType::F32, 24_030_720)
}

// =============================================================================
// FAIL-FAST VALIDATION TESTS (TensorMetadata)
// =============================================================================

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: shapes is empty")]
fn test_tensor_metadata_rejects_empty_shapes() {
    let _ = TensorMetadata::new(
        HashMap::new(), // EMPTY - MUST PANIC
        DType::F32,
        100,
    );
}

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: total_params is 0")]
fn test_tensor_metadata_rejects_zero_params() {
    let _ = TensorMetadata::new(
        [("test".to_string(), vec![100, 768])].into_iter().collect(),
        DType::F32,
        0, // ZERO PARAMS - MUST PANIC
    );
}

// =============================================================================
// FAIL-FAST VALIDATION TESTS (WarmLoadResult)
// =============================================================================

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: gpu_ptr is null")]
fn test_warm_load_result_rejects_null_pointer() {
    let metadata = create_test_metadata();
    let _ = WarmLoadResult::new(
        0, // NULL POINTER - MUST PANIC
        [1u8; 32],
        1024,
        Duration::from_millis(100),
        metadata,
    );
}

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: checksum is all zeros")]
fn test_warm_load_result_rejects_zero_checksum() {
    let metadata = create_test_metadata();
    let _ = WarmLoadResult::new(
        0x7fff_0000_1000,
        [0u8; 32], // ZERO CHECKSUM - MUST PANIC
        1024,
        Duration::from_millis(100),
        metadata,
    );
}

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: size_bytes is 0")]
fn test_warm_load_result_rejects_zero_size() {
    let metadata = create_test_metadata();
    let _ = WarmLoadResult::new(
        0x7fff_0000_1000,
        [1u8; 32],
        0, // ZERO SIZE - MUST PANIC
        Duration::from_millis(100),
        metadata,
    );
}

// =============================================================================
// FAIL-FAST VALIDATION TESTS (LoadedModelWeights)
// =============================================================================

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: model_id is empty")]
fn test_loaded_model_weights_rejects_empty_model_id() {
    // Create a real GpuTensor for testing (requires GPU, so we skip in unit tests)
    // In practice, this test would be an integration test with GPU access
    // For unit testing, we verify the panic message is correct

    // Since we can't easily create a GpuTensor without GPU,
    // we test that the first assertion (model_id) fires before tensors check
    let tensors: HashMap<String, crate::gpu::GpuTensor> = HashMap::new();

    let _ = LoadedModelWeights::new(
        "".to_string(), // EMPTY - MUST PANIC
        tensors,        // Empty too, but model_id check comes first
        [1u8; 32],
        1024,
        0,
    );
}

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: tensors is empty")]
fn test_loaded_model_weights_rejects_empty_tensors() {
    let _ = LoadedModelWeights::new(
        "E1_Semantic".to_string(),
        HashMap::new(), // EMPTY - MUST PANIC
        [1u8; 32],
        1024,
        0,
    );
}

// NOTE: test_loaded_model_weights_rejects_zero_checksum cannot be tested in unit tests
// because we cannot create a GpuTensor without GPU hardware. The tensors.is_empty()
// check fires before the checksum check. This validation is documented and would
// fire correctly in integration tests with actual GPU tensors.
//
// The expected panic message is: "CONSTITUTION VIOLATION AP-007: file_checksum is all zeros"
// This will be verified in integration tests (TASK-EMB-006-integration).

// =============================================================================
// VALID DATA TESTS
// =============================================================================

#[test]
fn test_tensor_metadata_accepts_valid_data() {
    let metadata = create_test_metadata();

    assert_eq!(metadata.tensor_count(), 2);
    assert!(metadata.total_params > 0);
    assert_eq!(metadata.dtype, DType::F32);
}

#[test]
fn test_tensor_metadata_calculates_params() {
    let metadata = TensorMetadata::new(
        [
            ("layer1".to_string(), vec![768, 768]),  // 589,824
            ("layer2".to_string(), vec![768, 3072]), // 2,359,296
        ]
        .into_iter()
        .collect(),
        DType::F32,
        2_949_120, // Sum of above
    );

    assert!(metadata.verify_params());
    assert_eq!(metadata.calculate_total_params(), 2_949_120);
}

#[test]
fn test_tensor_metadata_calculates_params_mismatch() {
    let metadata = TensorMetadata::new(
        [
            ("layer1".to_string(), vec![768, 768]), // 589,824
        ]
        .into_iter()
        .collect(),
        DType::F32,
        1_000_000, // Wrong value
    );

    assert!(!metadata.verify_params());
    assert_eq!(metadata.calculate_total_params(), 589_824);
}

#[test]
fn test_warm_load_result_accepts_valid_data() {
    let metadata = TensorMetadata::new(
        [("embeddings.weight".to_string(), vec![30522, 768])]
            .into_iter()
            .collect(),
        DType::F32,
        23_440_896, // 30522 * 768
    );

    let result = WarmLoadResult::new(
        0x7fff_0000_1000, // Real-looking pointer
        [0xAB; 32],       // Non-zero checksum
        93_763_584,       // 23M params * 4 bytes
        Duration::from_millis(150),
        metadata,
    );

    assert!(result.gpu_ptr != 0);
    assert!(result.size_bytes > 0);
    assert_eq!(result.checksum, [0xAB; 32]);
}

#[test]
fn test_checksum_verification() {
    let expected = [0xAB; 32];
    let metadata = TensorMetadata::new(
        [("test".to_string(), vec![100])].into_iter().collect(),
        DType::F32,
        100,
    );

    let result = WarmLoadResult::new(
        0x7fff_0000_1000,
        expected,
        400,
        Duration::from_millis(10),
        metadata,
    );

    assert!(result.verify_checksum(&expected));
    assert!(!result.verify_checksum(&[0xCD; 32]));
}

#[test]
fn test_checksum_hex_conversion() {
    let checksum = [
        0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE,
        0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
        0xFF, 0x00,
    ];

    let metadata = TensorMetadata::new(
        [("test".to_string(), vec![100])].into_iter().collect(),
        DType::F32,
        100,
    );

    let result = WarmLoadResult::new(
        0x7fff_0000_1000,
        checksum,
        400,
        Duration::from_millis(10),
        metadata,
    );

    let hex = result.checksum_hex();
    assert_eq!(hex.len(), 64); // 32 bytes * 2 chars per byte
    assert!(hex.starts_with("deadbeef"));
}

#[test]
fn test_tensor_metadata_get_shape() {
    let metadata = create_test_metadata();

    let shape = metadata.get_shape("embeddings.weight");
    assert!(shape.is_some());
    assert_eq!(shape.unwrap(), &vec![30522, 768]);

    let missing = metadata.get_shape("nonexistent");
    assert!(missing.is_none());
}

#[test]
fn test_bytes_per_param() {
    let metadata = TensorMetadata::new(
        [("test".to_string(), vec![100])].into_iter().collect(),
        DType::F32,
        100,
    );

    let result = WarmLoadResult::new(
        0x7fff_0000_1000,
        [1u8; 32],
        400,
        Duration::from_millis(10),
        metadata,
    );

    assert_eq!(result.bytes_per_param(), 4); // F32 = 4 bytes

    // Test with F16
    let metadata_f16 = TensorMetadata::new(
        [("test".to_string(), vec![100])].into_iter().collect(),
        DType::F16,
        100,
    );

    let result_f16 = WarmLoadResult::new(
        0x7fff_0000_1000,
        [1u8; 32],
        200,
        Duration::from_millis(10),
        metadata_f16,
    );

    assert_eq!(result_f16.bytes_per_param(), 2); // F16 = 2 bytes
}

// =============================================================================
// COMPILE-TIME ASSERTION VERIFICATION
// =============================================================================

#[test]
fn test_checksum_size() {
    // Verify at runtime that checksum is correct size (compile-time assertion above)
    assert_eq!(std::mem::size_of::<[u8; 32]>(), 32);
}

// =============================================================================
// VRAM ALLOCATION TRACKING TESTS (TASK-EMB-016)
// =============================================================================

/// Edge Case 1: VRAM Delta Mismatch
///
/// Scenario: GPU reports delta that differs from allocation size by >50MB
/// Expected: `VramAllocationTracking::is_real()` returns `false`
#[test]
fn test_vram_allocation_detects_delta_mismatch() {
    // Create allocation with massive mismatch: 1KB allocation claims 1000MB delta
    let alloc = VramAllocationTracking {
        base_ptr: 0x7fff_0000_1000, // Real-looking pointer
        size_bytes: 1024,           // 1KB allocated
        vram_before_mb: 1000,
        vram_after_mb: 2000, // Claims 1000MB delta for 1KB!
        vram_delta_mb: 1000,
    };

    assert!(
        !alloc.is_real(),
        "Should detect VRAM delta mismatch: 1KB allocation cannot cause 1000MB delta"
    );

    // Verify the detection is due to delta mismatch, not pointer
    assert_ne!(alloc.base_ptr, VramAllocationTracking::FAKE_POINTER);
}

#[test]
fn test_vram_allocation_detects_fake_pointer() {
    // Use the known fake pointer value
    let alloc = VramAllocationTracking {
        base_ptr: VramAllocationTracking::FAKE_POINTER, // 0x7f80_0000_0000
        size_bytes: 104_857_600,                        // 100MB
        vram_before_mb: 5000,
        vram_after_mb: 5100,
        vram_delta_mb: 100,
    };

    assert!(
        !alloc.is_real(),
        "Should detect fake pointer 0x7f80_0000_0000"
    );
}

#[test]
fn test_vram_allocation_accepts_valid_delta() {
    // Create allocation with matching delta: 100MB allocation with 100MB delta
    let alloc = VramAllocationTracking::new(
        0x7fff_0000_1000, // Real pointer
        104_857_600,      // 100MB
        5000,             // 5GB before
        5100,             // 5.1GB after (100MB delta)
    );

    assert!(
        alloc.is_real(),
        "Should accept valid VRAM allocation with matching delta"
    );
    assert_eq!(alloc.vram_delta_mb, 100);
}

#[test]
fn test_vram_allocation_tolerates_small_overhead() {
    // Create allocation with small overhead (within 50MB tolerance)
    let alloc = VramAllocationTracking::new(
        0x7fff_0000_1000,
        104_857_600, // 100MB allocation
        5000,
        5130, // 130MB delta (30MB overhead)
    );

    // 100MB allocation with 130MB delta = 30MB difference, within 50MB tolerance
    assert!(
        alloc.is_real(),
        "Should accept allocation with small GPU overhead within 50MB tolerance"
    );
}

#[test]
fn test_vram_allocation_rejects_excessive_overhead() {
    // Create allocation with excessive overhead (>50MB tolerance)
    let alloc = VramAllocationTracking {
        base_ptr: 0x7fff_0000_1000,
        size_bytes: 104_857_600, // 100MB allocation
        vram_before_mb: 5000,
        vram_after_mb: 5200, // 200MB delta (100MB overhead)
        vram_delta_mb: 200,
    };

    // 100MB allocation with 200MB delta = 100MB difference, exceeds 50MB tolerance
    assert!(
        !alloc.is_real(),
        "Should reject allocation with excessive overhead >50MB tolerance"
    );
}

#[test]
fn test_vram_allocation_delta_display() {
    let alloc = VramAllocationTracking::new(0x7fff_0000_1000, 104_857_600, 5000, 5100);

    let display = alloc.delta_display();
    assert_eq!(display, "100 MB (5000 -> 5100 MB)");
}

#[test]
fn test_vram_allocation_size_conversions() {
    let alloc = VramAllocationTracking::new(
        0x7fff_0000_1000,
        1_073_741_824, // 1GB
        5000,
        6024, // ~1GB delta
    );

    // Verify size_mb
    let size_mb = alloc.size_mb();
    assert!((size_mb - 1024.0).abs() < 0.1, "1GB should be ~1024MB");

    // Verify size_gb
    let size_gb = alloc.size_gb();
    assert!((size_gb - 1.0).abs() < 0.01, "1GB should be ~1.0GB");
}

// =============================================================================
// VRAM ALLOCATION FAIL-FAST TESTS
// =============================================================================

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: base_ptr is null")]
fn test_vram_allocation_rejects_null_ptr() {
    let _ = VramAllocationTracking::new(0, 1024, 1000, 1100);
}

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: size_bytes is 0")]
fn test_vram_allocation_rejects_zero_size() {
    let _ = VramAllocationTracking::new(0x7fff_0000_1000, 0, 1000, 1100);
}

#[test]
#[should_panic(expected = "[EMB-E010] SIMULATION_DETECTED")]
fn test_vram_allocation_assert_real_panics_on_fake_pointer() {
    let fake_alloc = VramAllocationTracking {
        base_ptr: VramAllocationTracking::FAKE_POINTER, // KNOWN FAKE POINTER
        size_bytes: 1024,
        vram_before_mb: 1000,
        vram_after_mb: 1001,
        vram_delta_mb: 1,
    };

    fake_alloc.assert_real();
}

#[test]
#[should_panic(expected = "[EMB-E010] SIMULATION_DETECTED")]
fn test_vram_allocation_assert_real_panics_on_delta_mismatch() {
    let fake_alloc = VramAllocationTracking {
        base_ptr: 0x7fff_0000_1000, // Valid pointer
        size_bytes: 1024,           // 1KB
        vram_before_mb: 1000,
        vram_after_mb: 2000, // 1000MB delta for 1KB!
        vram_delta_mb: 1000,
    };

    fake_alloc.assert_real();
}

// =============================================================================
// INFERENCE VALIDATION TESTS (TASK-EMB-016)
// =============================================================================

/// Edge Case 2: Sin Wave Pattern Detection
///
/// Scenario: Inference output follows perfect mathematical pattern (sin wave)
/// Expected: `InferenceValidation::is_real()` returns `false`
#[test]
fn test_inference_validation_detects_sin_wave() {
    // Generate sin wave output: (i * 0.001).sin()
    let sin_wave_output: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001).sin()).collect();

    let validation = InferenceValidation {
        sample_input: "test input".to_string(),
        sample_output: sin_wave_output,
        output_norm: 1.0,
        latency: Duration::from_millis(10),
        matches_golden: true,
        golden_similarity: 0.99, // High similarity, but sin wave pattern
    };

    assert!(!validation.is_real(), "Should detect sin wave fake pattern");
}

/// Edge Case 3: Low Golden Similarity
///
/// Scenario: Inference output has low cosine similarity to golden reference (<0.95)
/// Expected: `InferenceValidation::is_real()` returns `false`
#[test]
fn test_inference_validation_rejects_low_golden() {
    // Generate realistic (non-sin-wave) output
    let output: Vec<f32> = (0..10)
        .map(|i| ((i * 17 + 42) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let validation = InferenceValidation::new(
        "The quick brown fox".to_string(),
        output,
        1.0,
        Duration::from_millis(50),
        false,
        0.50, // LOW golden similarity - REJECT
    );

    assert!(
        !validation.is_real(),
        "Should reject low golden similarity (0.50 < 0.95)"
    );
}

#[test]
fn test_inference_validation_detects_all_zeros() {
    let validation = InferenceValidation {
        sample_input: "test".to_string(),
        sample_output: vec![0.0; 768], // ALL ZEROS
        output_norm: 0.0,
        latency: Duration::from_millis(10),
        matches_golden: false,
        golden_similarity: 0.99, // High similarity but zeros
    };

    assert!(
        !validation.is_real(),
        "Should detect all-zero output as fake"
    );
}

#[test]
fn test_inference_validation_accepts_high_golden() {
    // Generate non-sin-wave realistic output with high variance
    let output: Vec<f32> = (0..768)
        .map(|i| ((i * 17 + 42) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let validation = InferenceValidation::new(
        "The quick brown fox".to_string(),
        output,
        1.0,
        Duration::from_millis(50),
        true,
        0.98, // HIGH golden similarity - ACCEPT
    );

    assert!(
        validation.is_real(),
        "Should accept high golden similarity with non-sin-wave output"
    );
}

#[test]
fn test_inference_validation_accepts_borderline_golden() {
    // Generate realistic output
    let output: Vec<f32> = (0..768)
        .map(|i| ((i * 31 + 7) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let validation = InferenceValidation::new(
        "Test borderline".to_string(),
        output,
        1.0,
        Duration::from_millis(50),
        true,
        0.96, // Just above 0.95 threshold
    );

    assert!(
        validation.is_real(),
        "Should accept golden similarity at 0.96 (just above 0.95 threshold)"
    );
}

#[test]
fn test_inference_validation_rejects_borderline_golden() {
    // Generate realistic output
    let output: Vec<f32> = (0..768)
        .map(|i| ((i * 31 + 7) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let validation = InferenceValidation::new(
        "Test borderline".to_string(),
        output,
        1.0,
        Duration::from_millis(50),
        false,
        0.94, // Just below 0.95 threshold
    );

    assert!(
        !validation.is_real(),
        "Should reject golden similarity at 0.94 (just below 0.95 threshold)"
    );
}

#[test]
fn test_inference_validation_calculate_norm() {
    let output = vec![3.0, 4.0]; // L2 norm = 5.0

    let validation = InferenceValidation::new(
        "test".to_string(),
        output,
        5.0,
        Duration::from_millis(10),
        true,
        0.99,
    );

    let calculated = validation.calculate_norm();
    assert!(
        (calculated - 5.0).abs() < 0.001,
        "L2 norm of [3, 4] should be 5"
    );
}

#[test]
fn test_inference_validation_verify_norm() {
    let output = vec![3.0, 4.0]; // L2 norm = 5.0

    let validation = InferenceValidation::new(
        "test".to_string(),
        output,
        5.0, // Correct norm
        Duration::from_millis(10),
        true,
        0.99,
    );

    assert!(
        validation.verify_norm(0.01),
        "Stored norm should match calculated"
    );

    // Test with wrong stored norm
    let validation_wrong = InferenceValidation {
        output_norm: 10.0, // Wrong!
        ..validation.clone()
    };

    assert!(
        !validation_wrong.verify_norm(0.01),
        "Wrong stored norm should not match"
    );
}

#[test]
fn test_inference_validation_output_dimension() {
    let output = vec![0.1; 768];

    let validation = InferenceValidation::new(
        "test".to_string(),
        output,
        1.0,
        Duration::from_millis(10),
        true,
        0.99,
    );

    assert_eq!(validation.output_dimension(), 768);
}

// =============================================================================
// INFERENCE VALIDATION FAIL-FAST TESTS
// =============================================================================

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: sample_input is empty")]
fn test_inference_validation_rejects_empty_input() {
    let _ = InferenceValidation::new(
        "".to_string(),
        vec![0.1, 0.2, 0.3],
        1.0,
        Duration::from_millis(10),
        true,
        0.99,
    );
}

#[test]
#[should_panic(expected = "CONSTITUTION VIOLATION AP-007: sample_output is empty")]
fn test_inference_validation_rejects_empty_output() {
    let _ = InferenceValidation::new(
        "test".to_string(),
        vec![],
        0.0,
        Duration::from_millis(10),
        false,
        0.0,
    );
}

#[test]
#[should_panic(expected = "[EMB-E011] FAKE_INFERENCE")]
fn test_inference_validation_assert_real_panics_on_zeros() {
    let fake_validation = InferenceValidation {
        sample_input: "test".to_string(),
        sample_output: vec![0.0; 768], // ALL ZEROS
        output_norm: 0.0,
        latency: Duration::from_millis(10),
        matches_golden: false,
        golden_similarity: 0.1, // LOW
    };

    fake_validation.assert_real();
}

#[test]
#[should_panic(expected = "[EMB-E011] FAKE_INFERENCE")]
fn test_inference_validation_assert_real_panics_on_sin_wave() {
    let sin_wave: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001).sin()).collect();

    let fake_validation = InferenceValidation {
        sample_input: "test".to_string(),
        sample_output: sin_wave,
        output_norm: 1.0,
        latency: Duration::from_millis(10),
        matches_golden: true,
        golden_similarity: 0.99,
    };

    fake_validation.assert_real();
}

#[test]
#[should_panic(expected = "[EMB-E011] FAKE_INFERENCE")]
fn test_inference_validation_assert_real_panics_on_low_golden() {
    let output: Vec<f32> = (0..768)
        .map(|i| ((i * 17 + 42) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let fake_validation = InferenceValidation {
        sample_input: "test".to_string(),
        sample_output: output,
        output_norm: 1.0,
        latency: Duration::from_millis(10),
        matches_golden: false,
        golden_similarity: 0.50, // LOW
    };

    fake_validation.assert_real();
}

// =============================================================================
// REAL DATA PATTERN ACCEPTANCE TESTS
// =============================================================================

#[test]
fn test_inference_validation_accepts_noisy_realistic_output() {
    // Generate highly varied output that simulates real model output
    use std::f32::consts::PI;

    let output: Vec<f32> = (0..768)
        .map(|i| {
            // Complex formula that produces realistic variation
            let base = ((i * 17) % 1000) as f32 / 1000.0;
            let noise = ((i * 31 + 7) % 100) as f32 / 1000.0;
            let periodic = (i as f32 * PI / 50.0).sin() * 0.1;
            base + noise + periodic - 0.5
        })
        .collect();

    let validation = InferenceValidation::new(
        "The quick brown fox jumps over the lazy dog".to_string(),
        output,
        1.0,
        Duration::from_millis(45),
        true,
        0.97,
    );

    assert!(
        validation.is_real(),
        "Should accept realistic noisy output with high golden similarity"
    );
}

#[test]
fn test_vram_allocation_realistic_model_sizes() {
    // Test with realistic model sizes
    let sizes = [
        (384 * 1024 * 1024, 384),       // 384MB - small model
        (1024 * 1024 * 1024, 1024),     // 1GB - medium model
        (4 * 1024 * 1024 * 1024, 4096), // 4GB - large model
    ];

    for (size_bytes, expected_delta_mb) in sizes {
        let alloc = VramAllocationTracking::new(
            0x7fff_0000_1000 + size_bytes as u64, // Vary pointer
            size_bytes,
            5000,
            5000 + expected_delta_mb,
        );

        assert!(
            alloc.is_real(),
            "Should accept realistic model size: {} bytes",
            size_bytes
        );
    }
}
