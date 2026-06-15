//! Basic constructor and accessor tests for SynergyMatrix.

use crate::teleological::synergy_matrix::{SynergyMatrix, SYNERGY_DIM};

#[test]
fn test_synergy_matrix_new() {
    let matrix = SynergyMatrix::new();

    // Diagonal should be 1.0
    for i in 0..SYNERGY_DIM {
        assert!(
            (matrix.values[i][i] - 1.0).abs() < f32::EPSILON,
            "Diagonal [{i}][{i}] should be 1.0"
        );
    }

    // Off-diagonal should be 0.0
    for i in 0..SYNERGY_DIM {
        for j in 0..SYNERGY_DIM {
            if i != j {
                assert!(
                    matrix.values[i][j].abs() < f32::EPSILON,
                    "Off-diagonal [{i}][{j}] should be 0.0"
                );
            }
        }
    }
}

#[test]
fn test_synergy_matrix_with_base_synergies() {
    let matrix = SynergyMatrix::with_base_synergies();

    // Verify diagonal is 1.0
    for i in 0..SYNERGY_DIM {
        assert!(
            (matrix.values[i][i] - 1.0).abs() < f32::EPSILON,
            "Diagonal [{i}][{i}] should be 1.0"
        );
    }

    // Verify some known synergy values from teleoplan.md
    assert!((matrix.get_synergy(0, 4) - 0.9).abs() < f32::EPSILON);
    assert!((matrix.get_synergy(1, 2) - 0.9).abs() < f32::EPSILON);
    assert!((matrix.get_synergy(5, 12) - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_synergy_matrix_symmetry() {
    let matrix = SynergyMatrix::with_base_synergies();

    for i in 0..SYNERGY_DIM {
        for j in 0..SYNERGY_DIM {
            assert!(
                (matrix.values[i][j] - matrix.values[j][i]).abs() < f32::EPSILON,
                "Matrix should be symmetric: [{i}][{j}] != [{j}][{i}]"
            );
        }
    }

    assert!(matrix.is_symmetric(f32::EPSILON));
}
