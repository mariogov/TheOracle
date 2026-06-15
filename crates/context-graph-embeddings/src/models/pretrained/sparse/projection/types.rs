//! Type definitions for sparse projection.
//!
//! Contains the ProjectionMatrix struct and associated constants.

use candle_core::{Device, Tensor};

use super::super::types::{SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};

/// Expected weight file path relative to model directory.
pub const PROJECTION_WEIGHT_FILE: &str = "sparse_projection.safetensors";

/// Expected tensor name in SafeTensors file.
pub const PROJECTION_TENSOR_NAME: &str = "projection.weight";

/// Learned projection matrix for sparse-to-dense conversion.
///
/// # Constitution Alignment
/// - E6_Sparse: `dim: "~30K 5%active"` input, 1536D output
/// - E13_Splade: Same architecture, same projection
///
/// # Weight Source
/// - Pre-trained via contrastive learning on MS MARCO
/// - Fine-tuned to preserve semantic similarity
///
/// # CRITICAL: No Fallback
/// If weight file is missing, system MUST panic. Hash fallback is FORBIDDEN (AP-007).
///
/// # Memory Layout
/// - Weight tensor: [30522, 1536] float32 = ~180MB on GPU
/// - Total VRAM requirement: ~180MB for weights only
#[derive(Debug)]
pub struct ProjectionMatrix {
    /// Weight tensor on GPU: [SPARSE_VOCAB_SIZE x SPARSE_PROJECTED_DIMENSION]
    /// Shape: [30522, 1536]
    pub(crate) weights: Tensor,

    /// Device where weights are loaded (must be CUDA for production)
    pub(crate) device: Device,

    /// SHA256 checksum of the weight file for integrity validation
    pub(crate) weight_checksum: [u8; 32],
}

#[allow(dead_code)]
impl ProjectionMatrix {
    /// Expected weight matrix shape: [vocab_size, projected_dim]
    /// Shape: [30522, 1536] per Constitution E6_Sparse
    pub const EXPECTED_SHAPE: (usize, usize) = (SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION);

    /// Expected file size in bytes: vocab_size * proj_dim * sizeof(f32)
    /// 30522 * 1536 * 4 = 187,527,168 bytes (~179MB)
    pub const EXPECTED_FILE_SIZE: usize = SPARSE_VOCAB_SIZE * SPARSE_PROJECTED_DIMENSION * 4;

    /// Get the weight tensor reference.
    ///
    /// # Returns
    /// Reference to the projection weight tensor [30522, 1536]
    #[inline]
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get the device where weights are stored.
    ///
    /// # Returns
    /// Reference to the Candle Device (should be CUDA in production)
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the weight file checksum for integrity verification.
    ///
    /// # Returns
    /// SHA256 checksum as 32-byte array
    #[inline]
    pub fn checksum(&self) -> &[u8; 32] {
        &self.weight_checksum
    }

    /// Check if weights are on a CUDA device.
    ///
    /// # Returns
    /// `true` if device is CUDA, `false` otherwise (e.g., CPU for testing)
    #[inline]
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Get the input dimension (vocabulary size).
    ///
    /// # Returns
    /// 30522 (BERT vocabulary size)
    #[inline]
    pub const fn input_dimension() -> usize {
        SPARSE_VOCAB_SIZE
    }

    /// Get the output dimension (projected dimension).
    ///
    /// # Returns
    /// 1536 per Constitution E6_Sparse
    #[inline]
    pub const fn output_dimension() -> usize {
        SPARSE_PROJECTED_DIMENSION
    }
}
