//! Query Result Tests

use context_graph_embeddings::{EmbedderQueryResult, MultiSpaceQueryResult};
use uuid::Uuid;

/// Test EmbedderQueryResult distance calculation.
#[test]
fn test_embedder_query_result_distance() {
    // Similarity 0.9 -> distance 0.1
    let result = EmbedderQueryResult::from_similarity(Uuid::new_v4(), 0, 0.9, 0);
    assert!(
        (result.distance - 0.1).abs() < f32::EPSILON,
        "Distance should be 1-similarity"
    );

    // Similarity 1.0 -> distance 0.0
    let result2 = EmbedderQueryResult::from_similarity(Uuid::new_v4(), 0, 1.0, 0);
    assert!(
        (result2.distance - 0.0).abs() < f32::EPSILON,
        "Distance for similarity 1.0 should be 0"
    );

    // Similarity 0.0 -> distance 1.0
    let result3 = EmbedderQueryResult::from_similarity(Uuid::new_v4(), 0, 0.0, 0);
    assert!(
        (result3.distance - 1.0).abs() < f32::EPSILON,
        "Distance for similarity 0.0 should be 1.0"
    );

    println!("[PASS] EmbedderQueryResult distance = 1 - similarity");
}

/// Test similarity clamping in distance calculation.
#[test]
fn test_similarity_clamping_in_distance() {
    // Similarity > 1.0 should clamp to 1.0
    let result = EmbedderQueryResult::from_similarity(Uuid::new_v4(), 0, 1.5, 0);
    assert!(
        (result.distance - 0.0).abs() < f32::EPSILON,
        "Clamped similarity 1.5->1.0 means distance 0"
    );

    // Similarity < -1.0 should clamp to -1.0
    let result2 = EmbedderQueryResult::from_similarity(Uuid::new_v4(), 0, -1.5, 0);
    assert!(
        (result2.distance - 2.0).abs() < f32::EPSILON,
        "Clamped similarity -1.5->-1.0 means distance 2.0"
    );

    println!("[PASS] Similarity clamping in distance calculation verified");
}

/// Test MultiSpaceQueryResult aggregation.
#[test]
fn test_multi_space_aggregation() {
    let id = Uuid::new_v4();

    let results = vec![
        EmbedderQueryResult::from_similarity(id, 0, 0.9, 0),
        EmbedderQueryResult::from_similarity(id, 5, 0.8, 1),
        EmbedderQueryResult::from_similarity(id, 12, 0.7, 2),
    ];

    let multi = MultiSpaceQueryResult::from_embedder_results(id, &results);

    // Verify embedder_count
    assert_eq!(multi.embedder_count, 3, "Should count 3 embedders");

    // Verify correct similarities are stored
    assert!((multi.embedder_similarities[0] - 0.9).abs() < f32::EPSILON);
    assert!((multi.embedder_similarities[5] - 0.8).abs() < f32::EPSILON);
    assert!((multi.embedder_similarities[12] - 0.7).abs() < f32::EPSILON);

    // Verify non-searched embedders are NaN
    assert!(
        multi.embedder_similarities[1].is_nan(),
        "Non-searched embedder should be NaN"
    );
    assert!(
        multi.embedder_similarities[6].is_nan(),
        "Non-searched embedder should be NaN"
    );

    // Verify weighted similarity = mean
    let expected_weighted = (0.9 + 0.8 + 0.7) / 3.0;
    assert!(
        (multi.weighted_similarity - expected_weighted).abs() < f32::EPSILON,
        "Weighted similarity should be mean: {}, got {}",
        expected_weighted,
        multi.weighted_similarity
    );

    println!("[PASS] MultiSpaceQueryResult aggregation verified");
}

/// Test MultiSpaceQueryResult panics on empty results.
#[test]
#[should_panic(expected = "AGGREGATION ERROR")]
fn test_multi_space_empty_results_panics() {
    let _ = MultiSpaceQueryResult::from_embedder_results(
        Uuid::new_v4(),
        &[], // Empty
    );
}
