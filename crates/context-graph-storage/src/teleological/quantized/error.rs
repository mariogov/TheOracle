//! Error types for quantized fingerprint storage.
//!
//! All errors include full context for debugging. Per FAIL FAST policy,
//! these errors are typically converted to panics at the call site.

use thiserror::Error;
use uuid::Uuid;

use crate::StorageError;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors specific to quantized fingerprint storage operations.
///
/// All errors include full context for debugging. Per FAIL FAST policy,
/// these errors are typically converted to panics at the call site.
#[derive(Debug, Error)]
pub enum QuantizedStorageError {
    /// Missing embedder in fingerprint (should have all 13).
    #[error(
        "STORAGE ERROR: Missing embedder {embedder_idx} for fingerprint {fingerprint_id}. \
         Expected {expected} embedders, found {found}. This indicates corrupted fingerprint."
    )]
    MissingEmbedder {
        fingerprint_id: Uuid,
        embedder_idx: u8,
        expected: usize,
        found: usize,
    },

    /// Column family not found in database.
    #[error(
        "STORAGE ERROR: Column family '{cf_name}' not found. \
         Database may need migration or was opened without quantized embedder CFs."
    )]
    ColumnFamilyNotFound { cf_name: String },

    /// Serialization failed.
    #[error(
        "STORAGE ERROR: Failed to serialize embedder {embedder_idx} for fingerprint {fingerprint_id}: {reason}"
    )]
    SerializationFailed {
        fingerprint_id: Uuid,
        embedder_idx: u8,
        reason: String,
    },

    /// Deserialization failed.
    #[error(
        "STORAGE ERROR: Failed to deserialize embedder {embedder_idx} for fingerprint {fingerprint_id}: {reason}"
    )]
    DeserializationFailed {
        fingerprint_id: Uuid,
        embedder_idx: u8,
        reason: String,
    },

    /// RocksDB write failed.
    #[error("STORAGE ERROR: RocksDB write failed for fingerprint {fingerprint_id}: {reason}")]
    WriteFailed {
        fingerprint_id: Uuid,
        reason: String,
    },

    /// RocksDB read failed.
    #[error("STORAGE ERROR: RocksDB read failed for fingerprint {fingerprint_id}: {reason}")]
    ReadFailed {
        fingerprint_id: Uuid,
        reason: String,
    },

    /// Fingerprint not found.
    #[error("STORAGE ERROR: Fingerprint {fingerprint_id} not found in storage.")]
    NotFound { fingerprint_id: Uuid },

    /// Version mismatch - no migration support per FAIL FAST.
    #[error(
        "STORAGE ERROR: Version mismatch for fingerprint {fingerprint_id}. \
         Expected version {expected}, found {found}. NO MIGRATION SUPPORT - data is incompatible."
    )]
    VersionMismatch {
        fingerprint_id: Uuid,
        expected: u8,
        found: u8,
    },

    /// Underlying storage error.
    #[error("STORAGE ERROR: {0}")]
    Storage(#[from] StorageError),
}

/// Result type for quantized storage operations.
pub type QuantizedStorageResult<T> = Result<T, QuantizedStorageError>;
