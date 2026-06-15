//! Hypervector operations for HDC.
//!
//! Core mathematical operations on hypervectors including:
//! - Bind (XOR): Associate two concepts
//! - Bundle (Majority): Combine multiple vectors
//! - Permute: Circular bit shift for positional encoding
//! - Hamming distance and similarity metrics

use super::types::{Hypervector, HDC_DIMENSION};
use bitvec::prelude::*;
use tracing::trace;

/// Binds two hypervectors using XOR.
///
/// XOR binding is:
/// - Commutative: A ^ B = B ^ A
/// - Self-inverse: A ^ B ^ B = A
/// - Approximately orthogonal to inputs
#[must_use]
pub fn bind(a: &Hypervector, b: &Hypervector) -> Hypervector {
    debug_assert_eq!(a.len(), HDC_DIMENSION);
    debug_assert_eq!(b.len(), HDC_DIMENSION);
    a.clone() ^ b.clone()
}

/// Bundles multiple hypervectors using majority vote.
///
/// For each bit position, the output is 1 if majority of inputs have 1.
/// Ties are broken by the first vector (deterministic).
///
/// # Arguments
/// * `vectors` - Slice of hypervectors to bundle
///
/// # Returns
/// Bundled hypervector, or zero vector if input is empty.
#[must_use]
pub fn bundle(vectors: &[Hypervector]) -> Hypervector {
    if vectors.is_empty() {
        return bitvec![u64, Lsb0; 0; HDC_DIMENSION];
    }

    if vectors.len() == 1 {
        return vectors[0].clone();
    }

    let mut result = bitvec![u64, Lsb0; 0; HDC_DIMENSION];
    let threshold = vectors.len() / 2;

    for i in 0..HDC_DIMENSION {
        let count: usize = vectors.iter().map(|v| v[i] as usize).sum();
        // Majority vote with tie-breaking by first vector
        if count > threshold || (count == threshold && vectors[0][i]) {
            result.set(i, true);
        }
    }

    trace!(
        num_vectors = vectors.len(),
        popcount = result.count_ones(),
        "Bundled hypervectors"
    );
    result
}

/// Permutes a hypervector by circular left shift.
///
/// Used for positional encoding: permute(v, n) represents position n.
///
/// # Arguments
/// * `hv` - Hypervector to permute
/// * `shift` - Number of positions to shift left
#[must_use]
pub fn permute(hv: &Hypervector, shift: usize) -> Hypervector {
    if shift == 0 || hv.is_empty() {
        return hv.clone();
    }

    let len = hv.len();
    let effective_shift = shift % len;

    let mut result = bitvec![u64, Lsb0; 0; len];
    for i in 0..len {
        let new_pos = (i + len - effective_shift) % len;
        result.set(new_pos, hv[i]);
    }
    result
}

/// Computes Hamming distance between two hypervectors.
///
/// Returns the number of bit positions where the vectors differ.
#[must_use]
pub fn hamming_distance(a: &Hypervector, b: &Hypervector) -> usize {
    debug_assert_eq!(a.len(), b.len());
    (a.clone() ^ b.clone()).count_ones()
}

/// Computes normalized similarity between two hypervectors.
///
/// Returns a value in [0, 1] where 1 = identical, 0.5 = orthogonal.
#[must_use]
pub fn similarity(a: &Hypervector, b: &Hypervector) -> f32 {
    let distance = hamming_distance(a, b);
    1.0 - (distance as f32 / HDC_DIMENSION as f32)
}
