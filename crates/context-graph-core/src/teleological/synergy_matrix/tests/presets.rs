//! Predefined constructor tests for SynergyMatrix (TASK-CORE-004).

use crate::teleological::synergy_matrix::{SynergyMatrix, SYNERGY_DIM};

#[test]
fn test_all_predefined_matrices_are_valid() {
    let matrices = [
        ("base", SynergyMatrix::with_base_synergies()),
        ("semantic_focused", SynergyMatrix::semantic_focused()),
        ("code_heavy", SynergyMatrix::code_heavy()),
        ("temporal_focused", SynergyMatrix::temporal_focused()),
        ("causal_reasoning", SynergyMatrix::causal_reasoning()),
        ("relational", SynergyMatrix::relational()),
        ("qualitative", SynergyMatrix::qualitative()),
        ("balanced", SynergyMatrix::balanced()),
        ("identity", SynergyMatrix::identity()),
    ];

    for (name, matrix) in matrices.iter() {
        let result = matrix.validate();
        assert!(
            result.is_ok(),
            "{} matrix validation failed: {:?}",
            name,
            result
        );
        assert!(
            matrix.is_valid(),
            "{} matrix is_valid() returned false",
            name
        );
        assert!(
            matrix.is_symmetric(f32::EPSILON),
            "{} matrix is not symmetric",
            name
        );
        assert!(
            matrix.has_unit_diagonal(f32::EPSILON),
            "{} matrix has non-unity diagonal",
            name
        );
        assert!(
            matrix.values_in_range(),
            "{} matrix has values out of range",
            name
        );
    }
}

#[test]
fn test_predefined_matrix_properties() {
    let semantic = SynergyMatrix::semantic_focused();
    let base = SynergyMatrix::with_base_synergies();

    // E1_Semantic row average should be higher in semantic_focused
    let semantic_e1_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 0)
        .map(|j| semantic.get_synergy(0, j))
        .sum::<f32>()
        / 12.0;
    let base_e1_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 0)
        .map(|j| base.get_synergy(0, j))
        .sum::<f32>()
        / 12.0;

    assert!(
        semantic_e1_avg > base_e1_avg,
        "semantic_focused E1 average ({}) should be higher than base ({})",
        semantic_e1_avg,
        base_e1_avg
    );

    let code = SynergyMatrix::code_heavy();

    // E6_Code row average should be higher in code_heavy
    let code_e6_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 5)
        .map(|j| code.get_synergy(5, j))
        .sum::<f32>()
        / 12.0;
    let base_e6_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 5)
        .map(|j| base.get_synergy(5, j))
        .sum::<f32>()
        / 12.0;

    assert!(
        code_e6_avg > base_e6_avg,
        "code_heavy E6 average ({}) should be higher than base ({})",
        code_e6_avg,
        base_e6_avg
    );
}
