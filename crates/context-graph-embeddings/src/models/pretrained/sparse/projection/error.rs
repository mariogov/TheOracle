//! Error types for sparse projection operations.
//!
//! # Error Codes (per SPEC-EMB-001)
//! - EMB-E001: CUDA_ERROR (GPU operation failed)
//! - EMB-E004: WEIGHT_CHECKSUM_MISMATCH (corrupted file)
//! - EMB-E005: DIMENSION_MISMATCH (wrong matrix shape)
//! - EMB-E006: PROJECTION_MATRIX_MISSING (file not found)
//! - EMB-E008: NOT_INITIALIZED (weights not loaded)

use std::path::PathBuf;

use thiserror::Error;

/// Errors that can occur during sparse projection.
///
/// # Error Codes (per SPEC-EMB-001)
/// - EMB-E001: CUDA_ERROR (GPU operation failed)
/// - EMB-E004: WEIGHT_CHECKSUM_MISMATCH (corrupted file)
/// - EMB-E005: DIMENSION_MISMATCH (wrong matrix shape)
/// - EMB-E006: PROJECTION_MATRIX_MISSING (file not found)
/// - EMB-E008: NOT_INITIALIZED (weights not loaded)
///
/// # Fail Fast Policy (Constitution AP-007)
/// All errors are non-recoverable. System MUST panic, NOT fall back to hash projection.
#[derive(Debug, Error)]
pub enum ProjectionError {
    /// Weight file not found at expected path.
    ///
    /// # Remediation
    /// Download from: https://huggingface.co/contextgraph/sparse-projection
    #[error(
        "[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found at {path}
  Expected: models/sparse_projection.safetensors
  Remediation: Download projection weights or train custom matrix"
    )]
    MatrixMissing { path: PathBuf },

    /// Weight file checksum does not match expected value.
    #[error(
        "[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
  Expected checksum: {expected}
  Actual checksum: {actual}
  File: {path}
  Remediation: Re-download weight file from trusted source"
    )]
    #[allow(dead_code)]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    /// Weight matrix has wrong shape.
    #[error(
        "[EMB-E005] DIMENSION_MISMATCH: Projection matrix has wrong shape
  Expected: [30522, 1536]
  Actual: [{actual_rows}, {actual_cols}]
  File: {path}
  Remediation: Ensure weight file matches BERT vocab (30522) to projection dim (1536)"
    )]
    DimensionMismatch {
        path: PathBuf,
        actual_rows: usize,
        actual_cols: usize,
    },

    /// GPU operation failed during projection.
    #[error(
        "[EMB-E001] CUDA_ERROR: GPU operation failed
  Operation: {operation}
  Details: {details}
  Remediation: Check GPU availability with nvidia-smi, verify driver version >= 545"
    )]
    GpuError { operation: String, details: String },

    /// Projection weights not loaded (must call load() first).
    #[error(
        "[EMB-E008] NOT_INITIALIZED: Projection weights not loaded
  Remediation: Call ProjectionMatrix::load() before calling project()"
    )]
    #[allow(dead_code)]
    NotInitialized,
}
