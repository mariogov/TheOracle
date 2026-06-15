//! Core dense vector similarity primitives.

use super::error::DenseSimilarityError;

/// Calculate L2 norm (magnitude) of a vector.
///
/// # Arguments
/// - `v`: The vector to compute norm for
///
/// # Returns
/// The L2 norm (Euclidean length) of the vector.
///
/// # Example
/// ```rust,ignore
/// let v = vec![3.0, 4.0];
/// let norm = l2_norm(&v);
/// assert!((norm - 5.0).abs() < 1e-6);
/// ```
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector to unit length in-place.
///
/// Does nothing if vector has zero magnitude (avoids division by zero).
///
/// # Arguments
/// - `v`: The vector to normalize in-place
///
/// # Example
/// ```rust,ignore
/// let mut v = vec![3.0, 4.0];
/// normalize(&mut v);
/// // v is now [0.6, 0.8]
/// ```
#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Internal dot product without validation.
/// Caller must ensure vectors have equal length.
#[inline]
pub(crate) fn dot_product_unchecked(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate dot product between two dense vectors.
///
/// # Arguments
/// - `a`: First vector
/// - `b`: Second vector
///
/// # Returns
/// The dot product of the two vectors.
///
/// # Errors
/// - `DenseSimilarityError::EmptyVector` if either vector is empty
/// - `DenseSimilarityError::DimensionMismatch` if vectors have different lengths
///
/// # Example
/// ```rust,ignore
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let dot = dot_product(&a, &b)?; // 32.0
/// ```
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(dot_product_unchecked(a, b))
}

/// Calculate cosine similarity between two dense vectors.
///
/// Returns value in [-1.0, 1.0] where 1.0 means identical direction,
/// 0.0 means orthogonal, and -1.0 means opposite direction.
///
/// # Arguments
/// - `a`: First vector
/// - `b`: Second vector
///
/// # Returns
/// Cosine similarity clamped to [-1.0, 1.0].
///
/// # Errors
/// - `DenseSimilarityError::EmptyVector` if either vector is empty
/// - `DenseSimilarityError::DimensionMismatch` if vectors have different lengths
/// - `DenseSimilarityError::ZeroMagnitude` if either vector has zero norm
///
/// # Example
/// ```rust,ignore
/// let a = vec![1.0, 0.0];
/// let b = vec![0.0, 1.0];
/// let sim = cosine_similarity(&a, &b)?; // 0.0 (orthogonal)
/// ```
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    let dot = dot_product_unchecked(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return Err(DenseSimilarityError::ZeroMagnitude);
    }

    let result = dot / (norm_a * norm_b);
    // Clamp to valid range to handle floating point errors
    Ok(result.clamp(-1.0, 1.0))
}

/// Calculate Euclidean distance between two dense vectors.
///
/// # Arguments
/// - `a`: First vector
/// - `b`: Second vector
///
/// # Returns
/// The Euclidean distance (L2 norm of difference).
///
/// # Errors
/// - `DenseSimilarityError::EmptyVector` if either vector is empty
/// - `DenseSimilarityError::DimensionMismatch` if vectors have different lengths
///
/// # Example
/// ```rust,ignore
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// let dist = euclidean_distance(&a, &b)?; // 5.0
/// ```
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Ok(sum.sqrt())
}
