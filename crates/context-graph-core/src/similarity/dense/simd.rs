//! SIMD-accelerated dense vector similarity (x86_64 AVX2).

use super::error::DenseSimilarityError;
use super::primitives::cosine_similarity;

/// Minimum vector length for SIMD to be beneficial (overhead vs. gain).
const SIMD_MIN_LENGTH: usize = 32;

/// Calculate cosine similarity using SIMD (AVX2 + FMA) instructions.
///
/// This function provides 2-4x speedup over the scalar implementation
/// for vectors with 256+ dimensions. For small vectors (<32 dims), it
/// falls back to the scalar implementation.
///
/// # Architecture Requirements
/// - x86_64 only
/// - Requires AVX2 and FMA instruction sets (runtime checked)
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
/// #[cfg(target_arch = "x86_64")]
/// {
///     let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
///     let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
///     let sim = cosine_similarity_simd(&a, &b)?;
/// }
/// ```
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    use std::arch::x86_64::*;

    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    // For small vectors, use scalar (SIMD overhead not worth it)
    if a.len() < SIMD_MIN_LENGTH {
        return cosine_similarity(a, b);
    }

    // Check AVX2 support at runtime
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return cosine_similarity(a, b);
    }

    // SAFETY: We have verified AVX2 and FMA support above, and we ensure
    // proper bounds checking throughout the implementation.
    unsafe {
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        let chunks = a.len() / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));

            // FMA: dot_sum += va * vb
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            // FMA: norm_a_sum += va * va
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            // FMA: norm_b_sum += vb * vb
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sum: reduce 8 lanes to 1
        let dot = hsum_avx(dot_sum);
        let norm_a_sq = hsum_avx(norm_a_sum);
        let norm_b_sq = hsum_avx(norm_b_sum);

        // Handle remainder with scalar code
        let remainder_start = chunks * 8;
        let mut dot_rem = 0.0f32;
        let mut norm_a_rem = 0.0f32;
        let mut norm_b_rem = 0.0f32;
        for i in remainder_start..a.len() {
            dot_rem += a[i] * b[i];
            norm_a_rem += a[i] * a[i];
            norm_b_rem += b[i] * b[i];
        }

        let total_dot = dot + dot_rem;
        let total_norm_a = (norm_a_sq + norm_a_rem).sqrt();
        let total_norm_b = (norm_b_sq + norm_b_rem).sqrt();

        if total_norm_a < f32::EPSILON || total_norm_b < f32::EPSILON {
            return Err(DenseSimilarityError::ZeroMagnitude);
        }

        let result = total_dot / (total_norm_a * total_norm_b);
        Ok(result.clamp(-1.0, 1.0))
    }
}

/// Horizontal sum of 8 f32 lanes in AVX register.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[inline]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Sum pairs: [a+b, c+d, e+f, g+h, a+b, c+d, e+f, g+h]
    let sum1 = _mm256_hadd_ps(v, v);
    // Sum pairs again: [a+b+c+d, e+f+g+h, a+b+c+d, e+f+g+h, ...]
    let sum2 = _mm256_hadd_ps(sum1, sum1);
    // Extract low and high 128-bit lanes and add
    let low = _mm256_extractf128_ps(sum2, 0);
    let high = _mm256_extractf128_ps(sum2, 1);
    let sum3 = _mm_add_ps(low, high);
    _mm_cvtss_f32(sum3)
}
