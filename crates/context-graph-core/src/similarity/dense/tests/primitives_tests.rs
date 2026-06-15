//! Tests for primitive similarity functions: cosine, dot product, euclidean, L2 norm, normalize.

use super::super::*;

// =============================================================================
// Cosine Similarity Tests
// =============================================================================

#[test]
fn test_cosine_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sim = cosine_similarity(&v, &v).unwrap();
    assert!(
        (sim - 1.0).abs() < 1e-6,
        "Identical vectors should have similarity 1.0, got {}",
        sim
    );
    println!(
        "[PASS] Cosine of identical vectors = 1.0: actual = {:.6}",
        sim
    );
}

#[test]
fn test_cosine_orthogonal_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!(
        sim.abs() < 1e-6,
        "Orthogonal vectors should have similarity 0.0, got {}",
        sim
    );
    println!(
        "[PASS] Cosine of orthogonal vectors = 0.0: actual = {:.6}",
        sim
    );
}

#[test]
fn test_cosine_opposite_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!(
        (sim + 1.0).abs() < 1e-6,
        "Opposite vectors should have similarity -1.0, got {}",
        sim
    );
    println!(
        "[PASS] Cosine of opposite vectors = -1.0: actual = {:.6}",
        sim
    );
}

#[test]
fn test_dimension_mismatch_error() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = cosine_similarity(&a, &b);
    assert!(matches!(
        result,
        Err(DenseSimilarityError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    ));
    println!(
        "[PASS] Dimension mismatch correctly detected: {:?}",
        result.err()
    );
}

#[test]
fn test_empty_vector_error() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0, 2.0];
    let result = cosine_similarity(&a, &b);
    assert!(matches!(result, Err(DenseSimilarityError::EmptyVector)));
    println!("[PASS] Empty vector correctly detected: {:?}", result.err());
}

#[test]
fn test_zero_magnitude_error() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = cosine_similarity(&a, &b);
    assert!(matches!(result, Err(DenseSimilarityError::ZeroMagnitude)));
    println!(
        "[PASS] Zero magnitude correctly detected: {:?}",
        result.err()
    );
}

// =============================================================================
// Dot Product Tests
// =============================================================================

#[test]
fn test_dot_product() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dot = dot_product(&a, &b).unwrap();
    let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0
    assert!((dot - expected).abs() < 1e-6);
    println!("[PASS] Dot product = {}, expected = {}", dot, expected);
}

#[test]
fn test_dot_product_empty_vector() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0, 2.0];
    let result = dot_product(&a, &b);
    assert!(matches!(result, Err(DenseSimilarityError::EmptyVector)));
    println!("[PASS] Dot product empty vector error: {:?}", result.err());
}

#[test]
fn test_dot_product_dimension_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = dot_product(&a, &b);
    assert!(matches!(
        result,
        Err(DenseSimilarityError::DimensionMismatch { .. })
    ));
    println!(
        "[PASS] Dot product dimension mismatch error: {:?}",
        result.err()
    );
}

// =============================================================================
// Euclidean Distance Tests
// =============================================================================

#[test]
fn test_euclidean_distance() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let dist = euclidean_distance(&a, &b).unwrap();
    assert!(
        (dist - 5.0).abs() < 1e-6,
        "Expected distance 5.0, got {}",
        dist
    );
    println!("[PASS] Euclidean distance = {}, expected = 5.0", dist);
}

#[test]
fn test_euclidean_distance_same_point() {
    let a = vec![1.0, 2.0, 3.0];
    let dist = euclidean_distance(&a, &a).unwrap();
    assert!(
        dist.abs() < 1e-6,
        "Distance to self should be 0.0, got {}",
        dist
    );
    println!("[PASS] Euclidean distance to self = {}", dist);
}

#[test]
fn test_euclidean_distance_empty_vector() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0, 2.0];
    let result = euclidean_distance(&a, &b);
    assert!(matches!(result, Err(DenseSimilarityError::EmptyVector)));
    println!(
        "[PASS] Euclidean distance empty vector error: {:?}",
        result.err()
    );
}

#[test]
fn test_euclidean_distance_dimension_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = euclidean_distance(&a, &b);
    assert!(matches!(
        result,
        Err(DenseSimilarityError::DimensionMismatch { .. })
    ));
    println!(
        "[PASS] Euclidean distance dimension mismatch error: {:?}",
        result.err()
    );
}

// =============================================================================
// L2 Norm Tests
// =============================================================================

#[test]
fn test_l2_norm() {
    let v = vec![3.0, 4.0];
    let norm = l2_norm(&v);
    assert!((norm - 5.0).abs() < 1e-6);
    println!("[PASS] L2 norm of [3,4] = {}, expected = 5.0", norm);
}

#[test]
fn test_l2_norm_zero_vector() {
    let v = vec![0.0, 0.0, 0.0];
    let norm = l2_norm(&v);
    assert!(norm.abs() < 1e-6);
    println!("[PASS] L2 norm of zero vector = {}", norm);
}

#[test]
fn test_l2_norm_single_element() {
    let v = vec![5.0];
    let norm = l2_norm(&v);
    assert!((norm - 5.0).abs() < 1e-6);
    println!("[PASS] L2 norm of [5] = {}", norm);
}

// =============================================================================
// Normalize Tests
// =============================================================================

#[test]
fn test_normalize() {
    let mut v = vec![3.0, 4.0];
    normalize(&mut v);
    let norm = l2_norm(&v);
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "Normalized vector should have norm 1.0, got {}",
        norm
    );
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
    println!("[PASS] Normalized [3,4] = [{:.3}, {:.3}]", v[0], v[1]);
}

#[test]
fn test_normalize_zero_vector() {
    let mut v = vec![0.0, 0.0, 0.0];
    let original = v.clone();
    normalize(&mut v);
    // Zero vector should remain unchanged
    assert_eq!(v, original);
    println!("[PASS] Normalize zero vector unchanged");
}

#[test]
fn test_normalize_already_normalized() {
    let mut v = vec![1.0, 0.0];
    normalize(&mut v);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!(v[1].abs() < 1e-6);
    println!(
        "[PASS] Already normalized vector unchanged: [{:.3}, {:.3}]",
        v[0], v[1]
    );
}
