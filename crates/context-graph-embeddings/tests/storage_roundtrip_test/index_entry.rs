//! Index Entry Tests

use context_graph_embeddings::IndexEntry;
use uuid::Uuid;

/// Test IndexEntry creation with precomputed norm.
#[test]
fn test_index_entry_creation_with_norm() {
    // Classic 3-4-5 right triangle
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![3.0, 4.0]);

    assert!(
        (entry.norm - 5.0).abs() < f32::EPSILON,
        "Norm should be 5.0"
    );
    assert_eq!(entry.vector.len(), 2, "Vector should have 2 dimensions");
    assert_eq!(entry.embedder_idx, 0, "Embedder index should be 0");

    println!("[PASS] IndexEntry created with correct precomputed norm");
}

/// Test normalized vector computation.
#[test]
fn test_index_entry_normalized() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![3.0, 4.0]);
    let normalized = entry.normalized();

    assert!(
        (normalized[0] - 0.6).abs() < f32::EPSILON,
        "First component should be 0.6"
    );
    assert!(
        (normalized[1] - 0.8).abs() < f32::EPSILON,
        "Second component should be 0.8"
    );

    // Verify unit norm
    let unit_norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (unit_norm - 1.0).abs() < 1e-6,
        "Normalized vector should have unit norm"
    );

    println!("[PASS] Normalized vector computed correctly");
}

/// Test cosine similarity: identical vectors = 1.0.
#[test]
fn test_cosine_similarity_identical() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![1.0, 2.0, 3.0]);
    let query = vec![1.0, 2.0, 3.0];

    let sim = entry.cosine_similarity(&query);
    assert!(
        (sim - 1.0).abs() < 1e-6,
        "Identical vectors should have similarity 1.0, got {}",
        sim
    );

    println!("[PASS] Cosine similarity for identical vectors = 1.0");
}

/// Test cosine similarity: opposite vectors = -1.0.
#[test]
fn test_cosine_similarity_opposite() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![1.0, 0.0, 0.0]);
    let query = vec![-1.0, 0.0, 0.0];

    let sim = entry.cosine_similarity(&query);
    assert!(
        (sim - (-1.0)).abs() < 1e-6,
        "Opposite vectors should have similarity -1.0, got {}",
        sim
    );

    println!("[PASS] Cosine similarity for opposite vectors = -1.0");
}

/// Test cosine similarity: perpendicular vectors = 0.0.
#[test]
fn test_cosine_similarity_perpendicular() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![1.0, 0.0]);
    let query = vec![0.0, 1.0];

    let sim = entry.cosine_similarity(&query);
    assert!(
        sim.abs() < 1e-6,
        "Perpendicular vectors should have similarity 0.0, got {}",
        sim
    );

    println!("[PASS] Cosine similarity for perpendicular vectors = 0.0");
}

/// Test cosine similarity is always in valid range [-1, 1].
/// Note: Due to floating point precision, values may slightly exceed [-1,1],
/// so we allow a small epsilon tolerance.
#[test]
fn test_cosine_similarity_range() {
    // Test with various random-ish vectors
    let test_cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![1.5, -2.3, 4.1], vec![-0.7, 3.2, 1.8]),
        (vec![100.0, -50.0], vec![0.001, 0.002]),
        (vec![0.1, 0.1, 0.1], vec![100.0, 100.0, 100.0]),
    ];

    const EPSILON: f32 = 1e-6;

    for (v1, v2) in test_cases {
        let entry = IndexEntry::new(Uuid::new_v4(), 0, v1);
        let sim = entry.cosine_similarity(&v2);

        assert!(
            (-1.0 - EPSILON..=1.0 + EPSILON).contains(&sim),
            "Cosine similarity {} out of range [-1, 1] with epsilon tolerance",
            sim
        );
    }

    println!("[PASS] Cosine similarity always in range [-1.0, 1.0] (with epsilon)");
}

/// Test cosine similarity panics on dimension mismatch.
#[test]
#[should_panic(expected = "SIMILARITY ERROR")]
fn test_cosine_similarity_dimension_mismatch() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![1.0, 2.0, 3.0]);
    let query = vec![1.0, 2.0]; // Wrong dimension

    let _ = entry.cosine_similarity(&query);
}

/// Test zero vector handling in normalized.
#[test]
fn test_zero_vector_normalized() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![0.0, 0.0, 0.0]);
    let normalized = entry.normalized();

    assert!(
        normalized.iter().all(|&x| x == 0.0),
        "Zero vector should normalize to zero vector"
    );
    assert!(
        entry.norm.abs() < f32::EPSILON,
        "Zero vector should have zero norm"
    );

    println!("[PASS] Zero vector normalized correctly to zero vector");
}

/// Test zero vector cosine similarity returns 0.0.
#[test]
fn test_zero_vector_cosine_similarity() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![0.0, 0.0, 0.0]);
    let query = vec![1.0, 2.0, 3.0];

    let sim = entry.cosine_similarity(&query);
    assert_eq!(
        sim, 0.0,
        "Zero vector should have 0.0 similarity with any vector"
    );

    println!("[PASS] Zero vector cosine similarity = 0.0");
}
