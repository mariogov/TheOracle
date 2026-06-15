//! Advanced tests for teleological matrix search.
//!
//! Tests for collection search, clustering, centroids, and comprehensive comparison.

use crate::teleological::comparison_error::ComparisonValidationError;
use crate::teleological::matrix_search::{
    ComponentWeights, MatrixSearchConfig, TeleologicalMatrixSearch,
};

use super::basic::{make_test_vector, make_varied_test_vector};

#[test]
fn test_matrix_search_collection() {
    let search = TeleologicalMatrixSearch::new();
    let query = make_test_vector(0.8, 0.7);

    let candidates = vec![
        make_test_vector(0.8, 0.7),
        make_varied_test_vector(200),
        make_varied_test_vector(500),
    ];

    let results = search.search(&query, &candidates);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 0);
    assert!(
        results[0].1 >= results[1].1,
        "First result should have highest similarity"
    );
}

#[test]
fn test_matrix_search_with_threshold() {
    let config = MatrixSearchConfig {
        min_similarity: 0.5,
        ..Default::default()
    };
    let search = TeleologicalMatrixSearch::with_config(config);

    let query = make_test_vector(0.8, 0.7);
    let candidates = vec![make_test_vector(0.8, 0.7), make_test_vector(0.1, 0.1)];

    let results = search.search(&query, &candidates);

    assert!(!results.is_empty());
    for (_, sim) in &results {
        assert!(*sim >= 0.5, "All results should be above threshold");
    }
}

#[test]
fn test_comprehensive_comparison() {
    let search = TeleologicalMatrixSearch::new();
    let tv1 = make_test_vector(0.8, 0.6);
    let tv2 = make_test_vector(0.7, 0.5);

    let comp = search.comprehensive_comparison(&tv1, &tv2);

    assert!(comp.full.overall > 0.0);
    assert!(comp.topic_profile_only > 0.0);
    assert!(comp.correlations_only > 0.0);
    assert!(comp.groups_only > 0.0);
    assert!(!comp.per_group.is_empty());
    assert!(comp.per_embedder_pattern.iter().all(|&v| v > 0.0));
}

#[test]
fn test_component_weights_validation() {
    let mut weights = ComponentWeights::default();
    assert!(
        weights.validate().is_ok(),
        "Default weights should sum to 1.0"
    );
    assert!(weights.is_valid(), "Default weights should be valid");

    weights.topic_profile = 0.5;
    let err = weights.validate();
    assert!(err.is_err(), "Modified weights should not sum to 1.0");

    match err {
        Err(ComparisonValidationError::WeightsNotNormalized { actual_sum, .. }) => {
            assert!((actual_sum - 1.1).abs() < 0.01, "Sum should be ~1.1");
        }
        _ => panic!("Expected WeightsNotNormalized error"),
    }

    weights.normalize();
    assert!(
        weights.validate().is_ok(),
        "Normalized weights should sum to 1.0"
    );
    assert!(weights.is_valid(), "Normalized weights should be valid");

    let bad_weights = ComponentWeights {
        confidence: -0.5,
        ..Default::default()
    };
    let range_err = bad_weights.validate();
    assert!(range_err.is_err(), "Negative weight should fail validation");
    match range_err {
        Err(ComparisonValidationError::WeightOutOfRange {
            field_name, value, ..
        }) => {
            assert_eq!(field_name, "confidence");
            assert!((value - (-0.5)).abs() < f32::EPSILON);
        }
        _ => panic!("Expected WeightOutOfRange error"),
    }
}
