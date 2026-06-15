//! Helper functions for RocksDbTeleologicalStore.
//!
//! Contains utility functions for computing similarity and formatting.

/// Encode a byte slice as lowercase hexadecimal string.
///
/// Used in error messages for RocksDB keys that are raw byte arrays.
pub fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Compute cosine similarity between two dense vectors.
///
/// Returns values in [0, 1] via normalization: `(raw_cosine + 1) / 2`.
/// This maps cosine [-1, 1] to [0, 1] where 0.5 = orthogonal, 1.0 = identical.
///
/// STOR-10 NOTE: HNSW search paths use `hnsw_distance_to_similarity()` which
/// also normalizes to [0, 1] for consistency. Both formulas map cos_sim [-1,1] → [0,1].
///
/// # Panics
/// STG-06 FIX: Panics in ALL build modes if vectors have different dimensions.
/// Dimension mismatches indicate upstream embedding pipeline bugs that must
/// be caught immediately, not silently hidden by returning 0.0.
pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // STG-06 FIX: FAIL FAST in ALL build modes. Dimension mismatches are programming
    // bugs that must be caught immediately. Previously, release mode silently returned
    // 0.0, hiding upstream bugs and suppressing legitimate search results.
    assert_eq!(
        a.len(),
        b.len(),
        "FATAL: Cosine similarity dimension mismatch: vector A has {} dimensions, \
         vector B has {} dimensions. This indicates a bug in the embedding pipeline \
         or a cross-embedder comparison (AP-02 violation). Fix the caller.",
        a.len(),
        b.len()
    );

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    if denom < f32::EPSILON {
        0.0
    } else {
        // Normalize from [-1, 1] to [0, 1] to match core crate's cosine_similarity()
        (dot / denom + 1.0) / 2.0
    }
}

/// Convert HNSW cosine distance to normalized similarity in [0, 1].
///
/// Usearch cosine distance = `1.0 - cos_sim`, ranging [0, 2].
/// We normalize: `(2.0 - distance) / 2.0 = (cos_sim + 1) / 2`.
/// This matches `compute_cosine_similarity()` which also normalizes to [0, 1].
///
/// STOR-10 FIX: Previously used `1.0 - distance.min(1.0)` which mapped
/// cos_sim [0,1] → [0,1] but clipped cos_sim [-1,0) → 0.0. This formula
/// instead maps the full range cos_sim [-1,1] → [0,1], consistent with
/// `compute_cosine_similarity()`. The `min_similarity` threshold now means
/// the same thing regardless of whether results come from HNSW or direct computation.
#[inline]
pub fn hnsw_distance_to_similarity(distance: f32) -> f32 {
    ((2.0 - distance) / 2.0).clamp(0.0, 1.0)
}
