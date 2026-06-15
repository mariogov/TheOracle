//! Sparse vector similarity functions for E6/E13 (SPLADE) embeddings.
//!
//! These functions operate on `SparseVector` types with ~30K vocabulary
//! but typically only 100-1000 active dimensions (~5% sparsity).
//!
//! # Performance
//!
//! Merge-join on sorted indices gives O(n + m) complexity where n, m
//! are the number of non-zero entries.
//!
//! # Sparse Embedders
//!
//! | Embedder | Vocab Size | Typical Sparsity |
//! |----------|------------|------------------|
//! | E6       | 30,522     | ~5% (1500 nnz)   |
//! | E13      | 30,522     | ~5% (1500 nnz)   |

use crate::types::fingerprint::SparseVector;
use std::cmp::Ordering;
use thiserror::Error;

/// Errors from sparse vector similarity computation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SparseSimilarityError {
    /// Empty sparse vector provided.
    #[error("Empty sparse vector provided")]
    EmptyVector,

    /// Index exceeds vocabulary size.
    #[error("Invalid index {index} exceeds vocabulary size {vocab_size}")]
    IndexOutOfBounds { index: usize, vocab_size: usize },

    /// Indices not sorted or contain duplicates.
    #[error("Indices not sorted or contain duplicates")]
    UnsortedIndices,

    /// Indices and values length mismatch.
    #[error("Indices and values length mismatch: indices={indices_len}, values={values_len}")]
    LengthMismatch {
        indices_len: usize,
        values_len: usize,
    },
}

/// Calculate Jaccard similarity based on active dimension overlap.
///
/// Jaccard similarity measures the ratio of shared dimensions to total
/// unique dimensions: |A ∩ B| / |A ∪ B|
///
/// # Returns
///
/// - 1.0 if both vectors are empty (considered identical)
/// - 0.0 if one is empty and other is not
/// - Jaccard coefficient in [0.0, 1.0] otherwise
///
/// # Note
///
/// This ignores the actual values and only considers which indices are active.
/// Use for measuring vocabulary overlap between SPLADE vectors.
///
/// # Example
///
/// ```rust,ignore
/// let a = SparseVector::new(vec![1, 2, 3], vec![0.5, 0.5, 0.5]).unwrap();
/// let b = SparseVector::new(vec![2, 3, 4], vec![0.5, 0.5, 0.5]).unwrap();
/// // Intersection: {2, 3} = 2, Union: {1, 2, 3, 4} = 4
/// let jaccard = jaccard_similarity(&a, &b);
/// assert!((jaccard - 0.5).abs() < 1e-6); // 2/4 = 0.5
/// ```
pub fn jaccard_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
    // Handle empty cases
    if a.is_empty() && b.is_empty() {
        return 1.0; // Both empty = identical
    }
    if a.is_empty() || b.is_empty() {
        return 0.0; // One empty = no overlap
    }

    // Use merge-join to count intersection (indices are sorted)
    let mut intersection = 0usize;
    let mut i = 0;
    let mut j = 0;

    while i < a.indices.len() && j < b.indices.len() {
        match a.indices[i].cmp(&b.indices[j]) {
            Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }

    // Union = |A| + |B| - |A ∩ B|
    let union = a.indices.len() + b.indices.len() - intersection;

    if union == 0 {
        return 1.0; // Edge case: both empty after loop (shouldn't happen)
    }

    intersection as f32 / union as f32
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::fingerprint::SparseVector;

    // ========================================================================
    // JACCARD SIMILARITY TESTS
    // ========================================================================

    #[test]
    fn test_jaccard_identical() {
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
        let sim = jaccard_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have Jaccard 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();
        let b = SparseVector::new(vec![2, 3], vec![3.0, 4.0]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        assert_eq!(sim, 0.0, "No overlap should have Jaccard 0.0");
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = SparseVector::new(vec![1, 2, 3], vec![0.5, 0.5, 0.5]).unwrap();
        let b = SparseVector::new(vec![2, 3, 4], vec![0.5, 0.5, 0.5]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        // Intersection: {2, 3} = 2, Union: {1, 2, 3, 4} = 4
        assert!((sim - 0.5).abs() < 1e-6, "Expected 0.5, got {}", sim);
    }

    #[test]
    fn test_jaccard_subset() {
        let a = SparseVector::new(vec![1, 2], vec![0.5, 0.5]).unwrap();
        let b = SparseVector::new(vec![1, 2, 3, 4], vec![0.5, 0.5, 0.5, 0.5]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        // Intersection: {1, 2} = 2, Union: {1, 2, 3, 4} = 4
        assert!((sim - 0.5).abs() < 1e-6, "Expected 0.5, got {}", sim);
    }

    #[test]
    fn test_jaccard_empty() {
        let empty = SparseVector::empty();
        let non_empty = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();

        // Both empty = identical
        assert_eq!(jaccard_similarity(&empty, &empty), 1.0);
        // One empty = no overlap
        assert_eq!(jaccard_similarity(&empty, &non_empty), 0.0);
        assert_eq!(jaccard_similarity(&non_empty, &empty), 0.0);
    }

    #[test]
    fn test_jaccard_ignores_values() {
        // Jaccard only considers active dimensions, not values
        let a = SparseVector::new(vec![1, 2, 3], vec![0.1, 0.1, 0.1]).unwrap();
        let b = SparseVector::new(vec![1, 2, 3], vec![10.0, 10.0, 10.0]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same indices should have Jaccard 1.0 regardless of values, got {}",
            sim
        );
    }

    #[test]
    fn test_synthetic_known_jaccard() {
        // Input: a = {1, 2, 3}, b = {2, 3, 4, 5}
        // Intersection: {2, 3} = 2 elements
        // Union: {1, 2, 3, 4, 5} = 5 elements
        // Expected Jaccard: 2/5 = 0.4
        let a = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVector::new(vec![2, 3, 4, 5], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let jaccard = jaccard_similarity(&a, &b);

        assert!(
            (jaccard - 0.4).abs() < 1e-6,
            "Expected 0.4, got {}",
            jaccard
        );
    }
}
