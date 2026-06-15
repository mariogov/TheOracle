//! Analysis function tests for SynergyMatrix.

use crate::teleological::synergy_matrix::{SynergyMatrix, CROSS_CORRELATION_COUNT, SYNERGY_DIM};

#[test]
fn test_synergy_matrix_average_synergy() {
    let matrix = SynergyMatrix::with_base_synergies();
    let avg = matrix.average_synergy();

    // Average should be in (0, 1)
    assert!(avg > 0.0 && avg < 1.0);

    // Count unique pairs: 13 * 12 / 2 = 78
    // Verify by manual calculation
    let mut sum = 0.0f32;
    for i in 0..SYNERGY_DIM {
        for j in (i + 1)..SYNERGY_DIM {
            sum += matrix.values[i][j];
        }
    }
    let expected = sum / CROSS_CORRELATION_COUNT as f32;

    assert!((avg - expected).abs() < f32::EPSILON);
}

#[test]
fn test_synergy_matrix_high_synergy_pairs() {
    let matrix = SynergyMatrix::with_base_synergies();
    let high_pairs = matrix.high_synergy_pairs(0.9);

    // Should include (0, 4) = E1_Semantic + E5_Analogical
    assert!(high_pairs.contains(&(0, 4)));
    // Should include (1, 2) = E2_Episodic + E3_Temporal
    assert!(high_pairs.contains(&(1, 2)));
}

#[test]
fn test_synergy_matrix_serialization() {
    let matrix = SynergyMatrix::with_base_synergies();
    let json = serde_json::to_string(&matrix).unwrap();
    let deserialized: SynergyMatrix = serde_json::from_str(&json).unwrap();

    assert_eq!(matrix.sample_count, deserialized.sample_count);
    assert!((matrix.get_synergy(0, 4) - deserialized.get_synergy(0, 4)).abs() < f32::EPSILON);
}
