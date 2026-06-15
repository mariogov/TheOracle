//! Core implementation for ProjectionMatrix.
//!
//! Contains `load()` and `project()` methods.

use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};

use super::super::types::{SparseVector, SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};
use super::error::ProjectionError;
use super::types::{ProjectionMatrix, PROJECTION_TENSOR_NAME, PROJECTION_WEIGHT_FILE};

#[allow(dead_code)]
impl ProjectionMatrix {
    /// Load projection matrix from SafeTensors file.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing `sparse_projection.safetensors`
    ///
    /// # Returns
    /// * `Ok(Self)` - Loaded projection matrix on GPU
    /// * `Err(ProjectionError)` - If loading fails
    ///
    /// # Errors
    /// - `MatrixMissing` - File not found at `{model_dir}/sparse_projection.safetensors`
    /// - `DimensionMismatch` - Tensor shape is not [30522, 1536]
    /// - `GpuError` - CUDA device unavailable or tensor upload failed
    ///
    /// # CRITICAL: No Fallback Policy (Constitution AP-007)
    /// If the weight file is missing, this function returns an error.
    /// Hash-based projection fallback (`idx % 1536`) is FORBIDDEN.
    /// If CUDA is unavailable, this function returns an error.
    /// CPU fallback is FORBIDDEN.
    ///
    /// # Example
    /// ```rust,ignore
    /// let model_dir = Path::new("/models/sparse");
    /// let projection = ProjectionMatrix::load(model_dir)?;
    /// assert!(projection.is_cuda());
    /// ```
    pub fn load(model_dir: &Path) -> Result<Self, ProjectionError> {
        let weight_path = model_dir.join(PROJECTION_WEIGHT_FILE);

        // Step 1: Read file bytes (REAL file read, not simulation)
        let file_bytes = fs::read(&weight_path).map_err(|e| {
            tracing::error!(
                "[EMB-E006] Weight file not found: {:?}, error: {}",
                weight_path,
                e
            );
            ProjectionError::MatrixMissing {
                path: weight_path.clone(),
            }
        })?;

        tracing::info!("Read {} bytes from {:?}", file_bytes.len(), weight_path);

        // Step 2: Compute REAL SHA256 checksum (no fake/placeholder values)
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        let checksum: [u8; 32] = hasher.finalize().into();

        tracing::debug!(
            "Computed SHA256 checksum: {:02x}{:02x}{:02x}{:02x}...",
            checksum[0],
            checksum[1],
            checksum[2],
            checksum[3]
        );

        // Step 3: Parse SafeTensors format
        let tensors = SafeTensors::deserialize(&file_bytes).map_err(|e| {
            tracing::error!("[EMB-E001] SafeTensors parse failed: {}", e);
            ProjectionError::GpuError {
                operation: "SafeTensors::deserialize".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 4: Get the projection.weight tensor
        let tensor_view = tensors.tensor(PROJECTION_TENSOR_NAME).map_err(|e| {
            tracing::error!(
                "[EMB-E006] Tensor '{}' not found in SafeTensors file: {}",
                PROJECTION_TENSOR_NAME,
                e
            );
            ProjectionError::MatrixMissing {
                path: weight_path.clone(),
            }
        })?;

        // Step 5: Validate shape is [30522, 1536]
        let shape = tensor_view.shape();
        if shape.len() != 2
            || shape[0] != SPARSE_VOCAB_SIZE
            || shape[1] != SPARSE_PROJECTED_DIMENSION
        {
            tracing::error!(
                "[EMB-E005] Shape mismatch: expected [{}, {}], got {:?}",
                SPARSE_VOCAB_SIZE,
                SPARSE_PROJECTED_DIMENSION,
                shape
            );
            return Err(ProjectionError::DimensionMismatch {
                path: weight_path,
                actual_rows: shape.first().copied().unwrap_or(0),
                actual_cols: shape.get(1).copied().unwrap_or(0),
            });
        }

        tracing::info!(
            "Tensor shape validated: [{}, {}]",
            SPARSE_VOCAB_SIZE,
            SPARSE_PROJECTED_DIMENSION
        );

        // Step 6: Create CUDA device (NO CPU fallback)
        let device = Device::cuda_if_available(0).map_err(|e| {
            tracing::error!("[EMB-E001] CUDA device creation failed: {}", e);
            ProjectionError::GpuError {
                operation: "Device::cuda_if_available".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 7: VERIFY we got CUDA, not CPU (AP-007 compliance)
        if !matches!(&device, Device::Cuda(_)) {
            tracing::error!(
                "[EMB-E001] CUDA device required but got CPU. No CPU fallback allowed."
            );
            return Err(ProjectionError::GpuError {
                operation: "CUDA verification".to_string(),
                details:
                    "No CUDA device available. CPU fallback is FORBIDDEN per Constitution AP-007."
                        .to_string(),
            });
        }

        tracing::info!("CUDA device acquired successfully");

        // Step 8: Load tensor data to GPU
        let weights = Tensor::from_raw_buffer(
            tensor_view.data(),
            DType::F32,
            &[SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION],
            &device,
        )
        .map_err(|e| {
            tracing::error!("[EMB-E001] Tensor GPU upload failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::from_raw_buffer".to_string(),
                details: e.to_string(),
            }
        })?;

        tracing::info!(
            "Loaded projection matrix to GPU: {:?}, checksum prefix: {:02x}{:02x}{:02x}{:02x}",
            weights.shape(),
            checksum[0],
            checksum[1],
            checksum[2],
            checksum[3]
        );

        Ok(Self {
            weights,
            device,
            weight_checksum: checksum,
        })
    }

    /// Project sparse vector to dense representation.
    ///
    /// # Algorithm
    /// 1. Validate input dimension == 30522
    /// 2. Convert sparse indices/weights to dense tensor [1, 30522]
    /// 3. Matrix multiply: dense_out = sparse_tensor @ weights^T
    /// 4. L2 normalize result
    /// 5. Return 1536D vector
    ///
    /// # Arguments
    /// * `sparse` - Input sparse vector (must have dimension == SPARSE_VOCAB_SIZE)
    ///
    /// # Returns
    /// * `Ok(Vec<f32>)` - L2-normalized 1536D dense vector
    /// * `Err(ProjectionError)` - If projection fails
    ///
    /// # Errors
    /// - `DimensionMismatch` - If input dimension != 30522 or index out of bounds
    /// - `GpuError` - If GPU operation fails
    ///
    /// # CRITICAL: No Fallback Policy (Constitution AP-007)
    /// This method MUST NOT fall back to hash-based projection.
    /// If GPU operation fails, return error - do NOT use CPU fallback.
    pub fn project(&self, sparse: &SparseVector) -> Result<Vec<f32>, ProjectionError> {
        // Step 1: Validate input dimension
        if sparse.dimension != SPARSE_VOCAB_SIZE {
            tracing::error!(
                "[EMB-E005] Input dimension mismatch: expected {}, got {}",
                SPARSE_VOCAB_SIZE,
                sparse.dimension
            );
            return Err(ProjectionError::DimensionMismatch {
                path: std::path::PathBuf::from("<input>"),
                actual_rows: 1,
                actual_cols: sparse.dimension,
            });
        }

        // Step 2: Handle empty sparse vector (edge case)
        if sparse.indices.is_empty() {
            tracing::warn!("[EMB-E005] Empty sparse vector - no non-zero indices");
            // Return zero vector - L2 norm would be undefined
            return Ok(vec![0.0f32; SPARSE_PROJECTED_DIMENSION]);
        }

        // Step 3: Convert sparse to dense tensor on GPU
        // Create dense representation: [1, SPARSE_VOCAB_SIZE]
        let mut dense_input = vec![0.0f32; SPARSE_VOCAB_SIZE];
        for (&idx, &weight) in sparse.indices.iter().zip(sparse.weights.iter()) {
            if idx >= SPARSE_VOCAB_SIZE {
                tracing::error!(
                    "[EMB-E005] Index {} out of bounds (max {})",
                    idx,
                    SPARSE_VOCAB_SIZE - 1
                );
                return Err(ProjectionError::DimensionMismatch {
                    path: std::path::PathBuf::from("<input>"),
                    actual_rows: 1,
                    actual_cols: idx + 1,
                });
            }
            dense_input[idx] = weight;
        }

        // Step 4: Create tensor on device [1, 30522]
        let sparse_tensor = Tensor::from_vec(dense_input, (1, SPARSE_VOCAB_SIZE), &self.device)
            .map_err(|e| {
                tracing::error!("[EMB-E001] Failed to create input tensor: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::from_vec (input)".to_string(),
                    details: e.to_string(),
                }
            })?;

        // Step 5: Matrix multiply: [1, 30522] @ [30522, 1536] = [1, 1536]
        let dense_output = sparse_tensor.matmul(&self.weights).map_err(|e| {
            tracing::error!("[EMB-E001] Matrix multiplication failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::matmul".to_string(),
                details: e.to_string(),
            }
        })?;

        // Step 6: L2 normalize on GPU
        let squared = dense_output.sqr().map_err(|e| {
            tracing::error!("[EMB-E001] Tensor sqr failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sqr".to_string(),
                details: e.to_string(),
            }
        })?;

        let sum_squared = squared.sum_all().map_err(|e| {
            tracing::error!("[EMB-E001] Tensor sum_all failed: {}", e);
            ProjectionError::GpuError {
                operation: "Tensor::sum_all".to_string(),
                details: e.to_string(),
            }
        })?;

        let norm_scalar: f32 = sum_squared
            .sqrt()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Tensor sqrt failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::sqrt".to_string(),
                    details: e.to_string(),
                }
            })?
            .to_scalar()
            .map_err(|e| {
                tracing::error!("[EMB-E001] to_scalar failed: {}", e);
                ProjectionError::GpuError {
                    operation: "to_scalar".to_string(),
                    details: e.to_string(),
                }
            })?;

        // Avoid division by zero
        let normalized = if norm_scalar > 1e-10 {
            (dense_output / norm_scalar as f64).map_err(|e| {
                tracing::error!("[EMB-E001] Tensor division failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor division".to_string(),
                    details: e.to_string(),
                }
            })?
        } else {
            tracing::warn!("Near-zero norm detected, returning unnormalized output");
            dense_output
        };

        // Step 7: Copy result to CPU
        let result_vec: Vec<f32> = normalized
            .flatten_all()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Tensor flatten failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::flatten_all".to_string(),
                    details: e.to_string(),
                }
            })?
            .to_vec1()
            .map_err(|e| {
                tracing::error!("[EMB-E001] Tensor to_vec1 failed: {}", e);
                ProjectionError::GpuError {
                    operation: "Tensor::to_vec1".to_string(),
                    details: e.to_string(),
                }
            })?;

        // Step 8: Verify output dimension
        if result_vec.len() != SPARSE_PROJECTED_DIMENSION {
            tracing::error!(
                "[EMB-E005] Output dimension mismatch: expected {}, got {}",
                SPARSE_PROJECTED_DIMENSION,
                result_vec.len()
            );
            return Err(ProjectionError::DimensionMismatch {
                path: std::path::PathBuf::from("<output>"),
                actual_rows: 1,
                actual_cols: result_vec.len(),
            });
        }

        tracing::debug!(
            "Projected sparse vector: {} non-zero -> {}D (norm: {:.4})",
            sparse.nnz(),
            result_vec.len(),
            norm_scalar
        );

        Ok(result_vec)
    }
}
