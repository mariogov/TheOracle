//! Batch projection implementation for ProjectionMatrix.
//!
//! Contains `project_batch()` method for efficient batched GPU operations.

use candle_core::Tensor;

use super::super::types::{SparseVector, SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};
use super::error::ProjectionError;
use super::types::ProjectionMatrix;

#[allow(dead_code)]
impl ProjectionMatrix {
    /// Project a batch of sparse vectors to dense representations.
    ///
    /// More efficient than calling `project()` repeatedly due to batched GPU operations.
    ///
    /// # Arguments
    /// * `batch` - Slice of sparse vectors to project
    ///
    /// # Returns
    /// * `Ok(Vec<Vec<f32>>)` - Vector of L2-normalized 1536D dense vectors
    /// * `Err(ProjectionError)` - If any projection fails
    pub fn project_batch(&self, batch: &[SparseVector]) -> Result<Vec<Vec<f32>>, ProjectionError> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch.len();

        // Validate all input dimensions
        for (i, sparse) in batch.iter().enumerate() {
            if sparse.dimension != SPARSE_VOCAB_SIZE {
                tracing::error!(
                    "[EMB-E005] Batch item {} dimension mismatch: expected {}, got {}",
                    i,
                    SPARSE_VOCAB_SIZE,
                    sparse.dimension
                );
                return Err(ProjectionError::DimensionMismatch {
                    path: std::path::PathBuf::from(format!("<batch[{}]>", i)),
                    actual_rows: 1,
                    actual_cols: sparse.dimension,
                });
            }
        }

        // Convert all sparse vectors to dense matrix [batch_size, SPARSE_VOCAB_SIZE]
        let mut dense_batch = vec![0.0f32; batch_size * SPARSE_VOCAB_SIZE];
        for (row_idx, sparse) in batch.iter().enumerate() {
            let row_offset = row_idx * SPARSE_VOCAB_SIZE;
            for (&col_idx, &weight) in sparse.indices.iter().zip(sparse.weights.iter()) {
                if col_idx >= SPARSE_VOCAB_SIZE {
                    tracing::error!(
                        "[EMB-E005] Batch item {} index {} out of bounds",
                        row_idx,
                        col_idx
                    );
                    return Err(ProjectionError::DimensionMismatch {
                        path: std::path::PathBuf::from(format!("<batch[{}]>", row_idx)),
                        actual_rows: 1,
                        actual_cols: col_idx + 1,
                    });
                }
                dense_batch[row_offset + col_idx] = weight;
            }
        }

        // Create batch tensor on device [batch_size, 30522]
        let batch_tensor =
            Tensor::from_vec(dense_batch, (batch_size, SPARSE_VOCAB_SIZE), &self.device).map_err(
                |e| {
                    tracing::error!("[EMB-E001] Failed to create batch tensor: {}", e);
                    ProjectionError::GpuError {
                        operation: "Tensor::from_vec (batch)".to_string(),
                        details: e.to_string(),
                    }
                },
            )?;

        // Matrix multiply: [batch_size, 30522] @ [30522, 1536] = [batch_size, 1536]
        let output_tensor = batch_tensor.matmul(&self.weights).map_err(|e| {
            tracing::error!("[EMB-E001] Batch matmul failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::matmul (batch)".to_string(),
                details: e.to_string(),
            }
        })?;

        // L2 normalize each row
        let squared = output_tensor.sqr().map_err(|e| {
            tracing::error!("[EMB-E001] Batch sqr failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sqr (batch)".to_string(),
                details: e.to_string(),
            }
        })?;

        let sum_squared = squared.sum_keepdim(1).map_err(|e| {
            tracing::error!("[EMB-E001] Batch sum_keepdim failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sum_keepdim".to_string(),
                details: e.to_string(),
            }
        })?;

        let norms = sum_squared.sqrt().map_err(|e| {
            tracing::error!("[EMB-E001] Batch sqrt failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sqrt (batch)".to_string(),
                details: e.to_string(),
            }
        })?;

        // Clamp norms to avoid division by zero
        let norms_clamped = norms.clamp(1e-10, f64::MAX).map_err(|e| {
            tracing::error!("[EMB-E001] Batch clamp failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::clamp".to_string(),
                details: e.to_string(),
            }
        })?;

        // Broadcast divide: [batch_size, 1536] / [batch_size, 1]
        let normalized = output_tensor.broadcast_div(&norms_clamped).map_err(|e| {
            tracing::error!("[EMB-E001] Batch broadcast_div failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::broadcast_div".to_string(),
                details: e.to_string(),
            }
        })?;

        // Copy results to CPU
        let flat_results: Vec<f32> = normalized
            .flatten_all()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Batch flatten failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::flatten_all (batch)".to_string(),
                    details: e.to_string(),
                }
            })?
            .to_vec1()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Batch to_vec1 failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::to_vec1 (batch)".to_string(),
                    details: e.to_string(),
                }
            })?;

        // Split into individual vectors
        let results: Vec<Vec<f32>> = flat_results
            .chunks(SPARSE_PROJECTED_DIMENSION)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Verify dimensions
        if results.len() != batch_size {
            tracing::error!(
                "[EMB-E005] Batch output count mismatch: expected {}, got {}",
                batch_size,
                results.len()
            );
            return Err(ProjectionError::DimensionMismatch {
                path: std::path::PathBuf::from("<batch_output>"),
                actual_rows: results.len(),
                actual_cols: SPARSE_PROJECTED_DIMENSION,
            });
        }

        tracing::debug!(
            "Projected batch of {} sparse vectors to {}D each",
            batch_size,
            SPARSE_PROJECTED_DIMENSION
        );

        Ok(results)
    }
}
