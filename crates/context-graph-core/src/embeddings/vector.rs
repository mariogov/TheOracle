//! Generic vector types for the 13-space embedding system.
//!
//! This module provides DenseVector and BinaryVector types used by
//! distance calculators and similarity functions.
//!
//! Note: SparseVector is in types/fingerprint/sparse.rs (specialized for SPLADE).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for vector operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

// =============================================================================
// DenseVector
// =============================================================================

/// Dense vector of f32 values for embedding storage and similarity computation.
///
/// Used for E1-E5, E7-E11 embeddings (all dense embedders).
///
/// # Example
///
/// ```
/// use context_graph_core::embeddings::vector::DenseVector;
///
/// let a = DenseVector::new(vec![1.0, 0.0, 0.0]);
/// let b = DenseVector::new(vec![0.0, 1.0, 0.0]);
/// assert!(a.cosine_similarity(&b).abs() < 1e-6); // Orthogonal
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DenseVector {
    data: Vec<f32>,
}

impl DenseVector {
    /// Create a new dense vector from data.
    #[inline]
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Create a zero vector of specified dimension.
    #[inline]
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
        }
    }

    /// Number of dimensions.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty (zero dimensions).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable access to underlying data.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access to underlying data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Dot product with another vector.
    ///
    /// Assumes same dimension - mismatched dimensions produce truncated result.
    pub fn dot_product(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// L2 magnitude (Euclidean norm).
    pub fn magnitude(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Cosine similarity with another vector.
    ///
    /// Returns 0.0 if either vector has zero magnitude (AP-10 compliance).
    /// Output is clamped to [-1.0, 1.0] for numerical stability.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot_product(other);
        let mag_a = self.magnitude();
        let mag_b = other.magnitude();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0; // Handle zero vectors gracefully (AP-10)
        }

        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }

    /// Euclidean distance to another vector.
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize in place to unit magnitude.
    ///
    /// No-op if magnitude is zero.
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            for x in &mut self.data {
                *x /= mag;
            }
        }
    }

    /// Return a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }
}

// =============================================================================
// BinaryVector
// =============================================================================

/// Bit-packed binary vector for hyperdimensional computing.
///
/// Bits are packed into u64 words for efficient storage and XOR operations.
/// Used for Hamming distance calculations.
///
/// # Example
///
/// ```
/// use context_graph_core::embeddings::vector::BinaryVector;
///
/// let mut v = BinaryVector::zeros(128);
/// v.set_bit(0, true);
/// v.set_bit(64, true);
/// assert_eq!(v.popcount(), 2);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BinaryVector {
    data: Vec<u64>,
    bit_len: usize,
}

impl BinaryVector {
    /// Create from raw u64 words and bit length.
    ///
    /// The data vector should have ceil(bit_len / 64) elements.
    #[inline]
    pub fn new(data: Vec<u64>, bit_len: usize) -> Self {
        Self { data, bit_len }
    }

    /// Create a zero vector of specified bit length.
    #[inline]
    pub fn zeros(bit_len: usize) -> Self {
        let num_words = bit_len.div_ceil(64);
        Self {
            data: vec![0u64; num_words],
            bit_len,
        }
    }

    /// Number of bits in the vector.
    #[inline]
    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Storage size in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.data.len() * 8
    }

    /// Set a bit at the specified index.
    ///
    /// Silently ignores out-of-bounds indices.
    pub fn set_bit(&mut self, index: usize, value: bool) {
        if index >= self.bit_len {
            return;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if value {
            self.data[word_idx] |= 1u64 << bit_idx;
        } else {
            self.data[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Get bit value at the specified index.
    ///
    /// Returns false for out-of-bounds indices.
    pub fn get_bit(&self, index: usize) -> bool {
        if index >= self.bit_len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.data[word_idx] >> bit_idx) & 1 == 1
    }

    /// Count of set bits (population count).
    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|w| w.count_ones()).sum()
    }

    /// Hamming distance to another binary vector.
    ///
    /// Counts number of bit positions that differ.
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DenseVector Tests
    // =========================================================================

    #[test]
    fn test_dense_new_and_accessors() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!(!v.is_empty());
        assert_eq!(v.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_zeros() {
        let v = DenseVector::zeros(5);
        assert_eq!(v.len(), 5);
        assert_eq!(v.data(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dense_dot_product() {
        let a = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let b = DenseVector::new(vec![4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((a.dot_product(&b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_magnitude() {
        let v = DenseVector::new(vec![3.0, 4.0]);
        // sqrt(9 + 16) = 5
        assert!((v.magnitude() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_cosine_identical_normalized() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]).normalized();
        let sim = v.cosine_similarity(&v);
        assert!((sim - 1.0).abs() < 1e-6, "Expected 1.0, got {}", sim);
    }

    #[test]
    fn test_dense_cosine_orthogonal() {
        let a = DenseVector::new(vec![1.0, 0.0]);
        let b = DenseVector::new(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&b);
        assert!(sim.abs() < 1e-6, "Expected 0.0, got {}", sim);
    }

    #[test]
    fn test_dense_cosine_zero_vector() {
        let zero = DenseVector::zeros(3);
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        // Zero vector should return 0.0, NOT NaN (AP-10)
        assert_eq!(zero.cosine_similarity(&v), 0.0);
        assert_eq!(v.cosine_similarity(&zero), 0.0);
        assert_eq!(zero.cosine_similarity(&zero), 0.0);
    }

    #[test]
    fn test_dense_euclidean_identical() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.euclidean_distance(&v), 0.0);
    }

    #[test]
    fn test_dense_euclidean_3_4_triangle() {
        let a = DenseVector::new(vec![0.0, 0.0]);
        let b = DenseVector::new(vec![3.0, 4.0]);
        let dist = a.euclidean_distance(&b);
        assert!((dist - 5.0).abs() < 1e-6, "Expected 5.0, got {}", dist);
    }

    #[test]
    fn test_dense_normalize() {
        let mut v = DenseVector::new(vec![3.0, 4.0]);
        v.normalize();
        assert!((v.magnitude() - 1.0).abs() < 1e-6);
        assert!((v.data()[0] - 0.6).abs() < 1e-6);
        assert!((v.data()[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_dense_normalize_zero() {
        let mut zero = DenseVector::zeros(3);
        zero.normalize(); // Should not panic or produce NaN
        assert_eq!(zero.data(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dense_serialization_roundtrip() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let bytes = bincode::serialize(&v).expect("serialize failed");
        let restored: DenseVector = bincode::deserialize(&bytes).expect("deserialize failed");
        assert_eq!(v, restored);
    }

    #[test]
    fn test_dense_default() {
        let v = DenseVector::default();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_dense_data_mut() {
        let mut v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        v.data_mut()[0] = 10.0;
        assert_eq!(v.data(), &[10.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_negative_values() {
        let a = DenseVector::new(vec![1.0, -1.0]);
        let b = DenseVector::new(vec![-1.0, 1.0]);
        let sim = a.cosine_similarity(&b);
        // Opposite direction vectors should have cosine similarity -1.0
        assert!((sim - (-1.0)).abs() < 1e-6, "Expected -1.0, got {}", sim);
    }

    // =========================================================================
    // BinaryVector Tests
    // =========================================================================

    #[test]
    fn test_binary_zeros() {
        let v = BinaryVector::zeros(64);
        assert_eq!(v.bit_len(), 64);
        assert_eq!(v.byte_size(), 8);
        assert_eq!(v.popcount(), 0);
    }

    #[test]
    fn test_binary_set_get_bit() {
        let mut v = BinaryVector::zeros(128);
        assert!(!v.get_bit(0));
        v.set_bit(0, true);
        assert!(v.get_bit(0));
        v.set_bit(0, false);
        assert!(!v.get_bit(0));
    }

    #[test]
    fn test_binary_cross_word_boundary() {
        let mut v = BinaryVector::zeros(128);
        v.set_bit(63, true);
        v.set_bit(64, true);
        assert!(v.get_bit(63));
        assert!(v.get_bit(64));
        assert!(!v.get_bit(62));
        assert!(!v.get_bit(65));
    }

    #[test]
    fn test_binary_popcount() {
        let mut v = BinaryVector::zeros(128);
        v.set_bit(0, true);
        v.set_bit(64, true);
        v.set_bit(127, true);
        assert_eq!(v.popcount(), 3);
    }

    #[test]
    fn test_binary_hamming_identical() {
        let v = BinaryVector::zeros(64);
        assert_eq!(v.hamming_distance(&v), 0);
    }

    #[test]
    fn test_binary_hamming_distance() {
        let mut a = BinaryVector::zeros(64);
        let mut b = BinaryVector::zeros(64);
        // a = 011, b = 110 (bits 0,1 vs bits 1,2)
        a.set_bit(0, true);
        a.set_bit(1, true);
        b.set_bit(1, true);
        b.set_bit(2, true);
        // XOR = 101, hamming = 2
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_binary_out_of_bounds() {
        let mut v = BinaryVector::zeros(64);
        // Out of bounds set should be no-op
        v.set_bit(100, true);
        // Out of bounds get should return false
        assert!(!v.get_bit(100));
    }

    #[test]
    fn test_binary_serialization_roundtrip() {
        let mut v = BinaryVector::zeros(128);
        v.set_bit(42, true);
        v.set_bit(100, true);
        let bytes = bincode::serialize(&v).expect("serialize failed");
        let restored: BinaryVector = bincode::deserialize(&bytes).expect("deserialize failed");
        assert_eq!(v, restored);
        assert!(restored.get_bit(42));
        assert!(restored.get_bit(100));
    }

    #[test]
    fn test_binary_default() {
        let v = BinaryVector::default();
        assert_eq!(v.bit_len(), 0);
        assert_eq!(v.byte_size(), 0);
    }

    #[test]
    fn test_binary_new_with_data() {
        let data = vec![0b1010_u64, 0b0101_u64];
        let v = BinaryVector::new(data, 128);
        assert!(v.get_bit(1)); // bit 1 of first word is set (0b1010)
        assert!(v.get_bit(3)); // bit 3 of first word is set (0b1010)
        assert!(v.get_bit(64)); // bit 0 of second word is set (0b0101)
        assert!(v.get_bit(66)); // bit 2 of second word is set (0b0101)
    }

    #[test]
    fn test_binary_large_vector() {
        // Test a 1024-bit vector (typical E9 HDC size)
        let mut v = BinaryVector::zeros(1024);
        v.set_bit(0, true);
        v.set_bit(511, true);
        v.set_bit(512, true);
        v.set_bit(1023, true);
        assert_eq!(v.popcount(), 4);
        assert!(v.get_bit(0));
        assert!(v.get_bit(511));
        assert!(v.get_bit(512));
        assert!(v.get_bit(1023));
        assert!(!v.get_bit(1024)); // out of bounds
    }

    #[test]
    fn test_binary_all_ones() {
        let data = vec![u64::MAX; 2];
        let v = BinaryVector::new(data, 128);
        assert_eq!(v.popcount(), 128);
        for i in 0..128 {
            assert!(v.get_bit(i), "Bit {} should be set", i);
        }
    }
}
