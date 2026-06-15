//! Vector operations trait definition.

use async_trait::async_trait;

use crate::error::CudaResult;

/// GPU-accelerated vector operations.
///
/// Provides common operations for neural network and similarity search.
#[async_trait]
pub trait VectorOps: Send + Sync {
    /// Compute cosine similarity between two vectors.
    ///
    /// # Return Range
    ///
    /// **Raw cosine similarity in \[-1.0, 1.0\].**
    /// - `1.0` = identical direction
    /// - `0.0` = orthogonal
    /// - `-1.0` = opposite direction
    ///
    /// Callers needing \[0,1\] range must apply `(raw + 1.0) / 2.0` (SRC-3).
    async fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> CudaResult<f32>;

    /// Compute dot product of two vectors.
    async fn dot_product(&self, a: &[f32], b: &[f32]) -> CudaResult<f32>;

    /// Normalize a vector to unit length.
    async fn normalize(&self, v: &[f32]) -> CudaResult<Vec<f32>>;

    /// Batch cosine similarity: compare query against multiple vectors.
    ///
    /// # Return Range
    ///
    /// **Raw cosine similarity in \[-1.0, 1.0\]** per element.
    /// NOT normalized to \[0,1\]. Apply `(raw + 1.0) / 2.0` if needed (SRC-3).
    async fn batch_cosine_similarity(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> CudaResult<Vec<f32>>;

    /// Matrix multiplication for attention.
    async fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaResult<Vec<f32>>;

    /// Softmax activation.
    async fn softmax(&self, v: &[f32]) -> CudaResult<Vec<f32>>;

    /// Check if GPU acceleration is available.
    fn is_gpu_available(&self) -> bool;

    /// Get device name.
    fn device_name(&self) -> &str;
}
