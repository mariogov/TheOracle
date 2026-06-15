//! Storage error types.
//!
//! Defines error types for all storage operations.
//! Errors are designed for fail-fast debugging with descriptive messages.

use context_graph_core::types::{EdgeType, NodeId, ValidationError};
use thiserror::Error;

/// Storage operation errors.
///
/// Provides typed errors for all storage operations with descriptive messages.
/// Implements `std::error::Error` and `Display` via `thiserror`.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Database failed to open at the specified path.
    #[error("Failed to open database at '{path}': {message}")]
    OpenFailed {
        /// The path where database open was attempted
        path: String,
        /// The underlying error message from RocksDB
        message: String,
    },

    /// Column family not found in the database.
    #[error("Column family '{name}' not found")]
    ColumnFamilyNotFound {
        /// Name of the missing column family
        name: String,
    },

    /// Write operation failed.
    #[error("Write failed: {0}")]
    WriteFailed(String),

    /// Read operation failed.
    #[error("Read failed: {0}")]
    ReadFailed(String),

    /// Flush operation failed.
    #[error("Flush failed: {0}")]
    FlushFailed(String),

    /// Entity not found by ID.
    #[error("Node not found: {id}")]
    NotFound {
        /// The entity ID that was not found (formatted as string)
        id: String,
    },

    /// Serialization or deserialization failed.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Node validation failed before storage.
    #[error("Validation error: {0}")]
    ValidationFailed(String),

    /// Index corruption detected during scan or validation.
    #[error("Index corruption detected in {index_name}: {details}")]
    IndexCorrupted {
        /// Name of the corrupted index (e.g., "temporal", "tags", "sources")
        index_name: String,
        /// Details about what corruption was detected
        details: String,
    },

    /// Generic internal error for unexpected failures.
    #[error("Internal storage error: {0}")]
    Internal(String),
}

impl From<ValidationError> for StorageError {
    fn from(e: ValidationError) -> Self {
        StorageError::ValidationFailed(e.to_string())
    }
}

impl From<rocksdb::Error> for StorageError {
    fn from(e: rocksdb::Error) -> Self {
        StorageError::Internal(e.to_string())
    }
}

impl StorageError {
    /// Creates a `NotFound` error for a missing `MemoryNode`.
    pub fn not_found_node(id: NodeId) -> Self {
        StorageError::NotFound { id: id.to_string() }
    }

    /// Creates a `NotFound` error for a missing `GraphEdge`.
    pub fn not_found_edge(source: NodeId, target: NodeId, edge_type: EdgeType) -> Self {
        StorageError::NotFound {
            id: format!("{}->{}:{:?}", source, target, edge_type),
        }
    }

    /// Creates a `NotFound` error for a missing embedding.
    pub fn not_found_embedding(id: NodeId) -> Self {
        StorageError::NotFound {
            id: format!("embedding:{}", id),
        }
    }
}

/// Convenient Result type for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_open_failed() {
        let error = StorageError::OpenFailed {
            path: "/tmp/test".to_string(),
            message: "permission denied".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("/tmp/test"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn test_error_column_family_not_found() {
        let error = StorageError::ColumnFamilyNotFound {
            name: "unknown_cf".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("unknown_cf"));
    }

    #[test]
    fn test_error_write_failed() {
        let error = StorageError::WriteFailed("disk full".to_string());
        assert!(error.to_string().contains("disk full"));
    }

    #[test]
    fn test_error_read_failed() {
        let error = StorageError::ReadFailed("io error".to_string());
        assert!(error.to_string().contains("io error"));
    }

    #[test]
    fn test_error_flush_failed() {
        let error = StorageError::FlushFailed("sync failed".to_string());
        assert!(error.to_string().contains("sync failed"));
    }

    #[test]
    fn test_error_not_found() {
        let error = StorageError::NotFound {
            id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Node not found"));
        assert!(msg.contains("550e8400"));
    }

    #[test]
    fn test_error_serialization() {
        let error = StorageError::Serialization("invalid msgpack".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Serialization error"));
        assert!(msg.contains("invalid msgpack"));
    }

    #[test]
    fn test_error_validation_failed() {
        let error = StorageError::ValidationFailed("importance out of range".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Validation error"));
        assert!(msg.contains("importance out of range"));
    }

    #[test]
    fn test_from_validation_error() {
        let val_error = ValidationError::InvalidEmbeddingDimension {
            expected: 1536,
            actual: 100,
        };
        let storage_error: StorageError = val_error.into();
        assert!(matches!(storage_error, StorageError::ValidationFailed(_)));
    }

    #[test]
    fn test_error_index_corrupted() {
        let error = StorageError::IndexCorrupted {
            index_name: "temporal".to_string(),
            details: "UUID parse failed".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("temporal"));
        assert!(msg.contains("UUID parse failed"));
        assert!(msg.contains("Index corruption"));
    }

    #[test]
    fn test_error_internal() {
        let error = StorageError::Internal("unexpected state".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Internal storage error"));
        assert!(msg.contains("unexpected state"));
    }

    #[test]
    fn test_not_found_node() {
        let id = uuid::Uuid::new_v4();
        let error = StorageError::not_found_node(id);
        let msg = error.to_string();
        assert!(msg.contains(&id.to_string()));
        assert!(msg.contains("Node not found"));
    }

    #[test]
    fn test_not_found_edge() {
        let source = uuid::Uuid::new_v4();
        let target = uuid::Uuid::new_v4();
        let error = StorageError::not_found_edge(source, target, EdgeType::Semantic);
        let msg = error.to_string();
        assert!(msg.contains(&source.to_string()[..8]));
        assert!(msg.contains(&target.to_string()[..8]));
        assert!(msg.contains("Semantic"));
    }

    #[test]
    fn test_not_found_embedding() {
        let id = uuid::Uuid::new_v4();
        let error = StorageError::not_found_embedding(id);
        let msg = error.to_string();
        assert!(msg.contains("embedding:"));
        assert!(msg.contains(&id.to_string()));
    }

    #[test]
    fn test_storage_result_type_alias() {
        fn returns_ok_result() -> StorageResult<String> {
            Ok("test".to_string())
        }

        fn returns_err_result() -> StorageResult<String> {
            Err(StorageError::Internal("test error".to_string()))
        }

        assert!(returns_ok_result().is_ok());
        assert!(returns_err_result().is_err());
    }

    #[test]
    fn test_error_debug() {
        let error = StorageError::WriteFailed("test".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("WriteFailed"));
    }
}
