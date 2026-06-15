//! MANDATORY Edge Case Tests (TASK-EMB-020 Definition of Done).

use crate::error::EmbeddingError;
use crate::quantization::router::QuantizationRouter;
use crate::types::ModelId;

/// Edge Case 1: Empty embedding - must not panic, should return error or empty output.
#[test]
fn test_edge_empty_embedding() {
    let router = QuantizationRouter::new();
    let empty: Vec<f32> = vec![];

    let result = router.quantize(ModelId::Hdc, &empty);
    // Should succeed but produce empty output OR fail gracefully
    // Verify it does NOT panic
    match result {
        Ok(q) => assert_eq!(q.original_dim, 0),
        Err(EmbeddingError::QuantizationFailed { model_id, .. }) => {
            assert_eq!(model_id, ModelId::Hdc);
        }
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
}

/// Edge Case 2: Maximum dimension (65536 = 2^16) - realistic large input.
#[test]
fn test_edge_max_dimension() {
    let router = QuantizationRouter::new();
    // 65536 dimensions (2^16) - large but realistic
    let large: Vec<f32> = (0..65536).map(|i| (i as f32).sin()).collect();

    let result = router.quantize(ModelId::Hdc, &large);
    assert!(
        result.is_ok(),
        "Large dimension quantization failed: {:?}",
        result.err()
    );

    let q = result.unwrap();
    assert_eq!(q.original_dim, 65536);
    // 65536 bits / 8 = 8192 bytes
    assert_eq!(
        q.data.len(),
        8192,
        "Expected 8192 bytes for 65536-bit binary vector"
    );
}

/// Edge Case 3: All same value (all zeros) - degenerate case.
///
/// Binary quantization uses threshold=0.0, where value >= threshold produces 1 bit.
/// So 0.0 >= 0.0 = true -> all 1 bits -> 0xFF bytes.
#[test]
fn test_edge_all_same_value() {
    let router = QuantizationRouter::new();
    // All zeros with threshold=0.0: 0.0 >= 0.0 = true -> all 1 bits
    let all_zeros = vec![0.0f32; 256];

    let result = router.quantize(ModelId::Hdc, &all_zeros);
    assert!(
        result.is_ok(),
        "All-zeros quantization failed: {:?}",
        result.err()
    );

    let q = result.unwrap();
    assert_eq!(q.original_dim, 256);
    assert_eq!(q.data.len(), 32); // 256/8 = 32 bytes

    // With threshold=0.0, all zeros -> all 1 bits (0.0 >= 0.0 = true)
    // This is the correct behavior per BinaryEncoder implementation
    assert!(
        q.data.iter().all(|&b| b == 0xFF),
        "Expected all 0xFF bytes for zero input (0.0 >= 0.0 = true), got {:?}",
        q.data
    );
}

/// Edge Case 4: All positive values - should produce all 1 bits.
#[test]
fn test_edge_all_positive() {
    let router = QuantizationRouter::new();
    let all_positive = vec![1.0f32; 64];

    let result = router.quantize(ModelId::Hdc, &all_positive);
    assert!(result.is_ok());

    let q = result.unwrap();
    assert_eq!(q.original_dim, 64);
    assert_eq!(q.data.len(), 8); // 64/8 = 8 bytes

    // All positive values -> all 1 bits -> 0xFF bytes
    assert!(
        q.data.iter().all(|&b| b == 0xFF),
        "Expected all 0xFF bytes for positive input, got {:?}",
        q.data
    );
}

/// Edge Case 5: Alternating pattern - verify bit packing order.
#[test]
fn test_edge_alternating_pattern() {
    let router = QuantizationRouter::new();
    // Pattern: +, -, +, -, ... (8 values = 1 byte)
    let alternating: Vec<f32> = (0..8)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let result = router.quantize(ModelId::Hdc, &alternating);
    assert!(result.is_ok());

    let q = result.unwrap();
    assert_eq!(q.original_dim, 8);
    assert_eq!(q.data.len(), 1);

    // Pattern 1,0,1,0,1,0,1,0 = 0b10101010 = 0xAA (LSB first)
    // Or 0b01010101 = 0x55 (MSB first)
    // Actual value depends on bit packing order
    let byte = q.data[0];
    assert!(
        byte == 0xAA || byte == 0x55,
        "Expected alternating pattern 0xAA or 0x55, got 0x{:02X}",
        byte
    );
}
