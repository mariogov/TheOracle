//! Top-level unified error type for context-graph library.

use thiserror::Error;

use super::sub_errors::{
    ConfigError, EmbeddingError, GpuError, IndexError, McpError, StorageError,
};

// ============================================================================
// TOP-LEVEL UNIFIED ERROR TYPE
// ============================================================================

/// Top-level unified error type for context-graph library.
///
/// All crate errors should be convertible to this type via `From` implementations.
/// Provides JSON-RPC error code mapping for MCP protocol responses.
///
/// # JSON-RPC Error Codes
///
/// Each error variant maps to a JSON-RPC error code:
/// - `-32600` to `-32603`: Standard JSON-RPC errors
/// - `-32001` to `-32007`: Context Graph specific errors
/// - `-32008`: INDEX_ERROR
/// - `-32009`: GPU_ERROR
///
/// # Recoverability
///
/// Errors are classified as recoverable or non-recoverable:
/// - Recoverable: Can be retried (e.g., rate limiting, model not loaded)
/// - Non-recoverable: Require intervention (e.g., corruption, config errors)
///
/// # Examples
///
/// ```rust
/// use context_graph_core::error::{ContextGraphError, EmbeddingError};
/// use context_graph_core::Embedder;
///
/// let err = ContextGraphError::Embedding(EmbeddingError::ModelNotLoaded(Embedder::Semantic));
/// assert_eq!(err.error_code(), -32005);
/// assert!(err.is_recoverable());
/// assert!(!err.is_critical());
/// ```
#[derive(Debug, Error)]
pub enum ContextGraphError {
    /// Embedding-related error.
    ///
    /// Covers model loading, generation, quantization, and dimension issues.
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    /// Storage-related error.
    ///
    /// Covers database operations, serialization, and data integrity.
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    /// Index-related error.
    ///
    /// Covers HNSW index operations, search failures, and corruption.
    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    /// Configuration error.
    ///
    /// Covers missing configs, invalid values, and parse failures.
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// GPU/CUDA error.
    ///
    /// Covers device initialization, memory, and kernel failures.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// MCP protocol error.
    ///
    /// Covers request validation, authorization, and protocol violations.
    #[error("MCP error: {0}")]
    Mcp(#[from] McpError),

    /// Validation error for input data.
    ///
    /// # When This Occurs
    ///
    /// - Field value out of allowed range
    /// - Invalid format for parameters
    /// - NaN or Infinity in numeric fields
    #[error("Validation error: {0}")]
    Validation(String),

    /// Internal error indicating a bug or system failure.
    ///
    /// # When This Occurs
    ///
    /// - Invariant violation detected
    /// - Unrecoverable state corruption
    /// - Resource exhaustion
    ///
    /// These errors indicate bugs and should be investigated.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl ContextGraphError {
    /// Get JSON-RPC error code for MCP responses.
    ///
    /// Maps to codes defined in `crates/context-graph-mcp/src/protocol.rs`.
    ///
    /// # Returns
    ///
    /// Negative i32 error code per JSON-RPC 2.0 specification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::error::{ContextGraphError, EmbeddingError, McpError};
    /// use context_graph_core::Embedder;
    ///
    /// let err = ContextGraphError::Embedding(EmbeddingError::EmptyInput);
    /// assert_eq!(err.error_code(), -32005);
    ///
    /// let err = ContextGraphError::Validation("bad input".to_string());
    /// assert_eq!(err.error_code(), -32602);
    /// ```
    #[inline]
    pub fn error_code(&self) -> i32 {
        match self {
            Self::Embedding(_) => -32005, // EMBEDDING_ERROR
            Self::Storage(_) => -32004,   // STORAGE_ERROR
            Self::Index(_) => -32008,     // INDEX_ERROR (new)
            Self::Config(_) => -32603,    // INTERNAL_ERROR (config is internal)
            Self::Gpu(_) => -32009,       // GPU_ERROR (new)
            Self::Mcp(e) => e.error_code(),
            Self::Validation(_) => -32602, // INVALID_PARAMS
            Self::Internal(_) => -32603,   // INTERNAL_ERROR
        }
    }

    /// Check if this error is recoverable via retry.
    ///
    /// Recoverable errors can potentially succeed if retried with:
    /// - Waiting for model to load
    /// - Retrying after backoff (rate limiting)
    /// - Retrying after garbage collection (OOM)
    /// - Retrying transaction
    ///
    /// # Returns
    ///
    /// `true` if retry might succeed, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::error::{ContextGraphError, EmbeddingError, StorageError};
    /// use context_graph_core::Embedder;
    ///
    /// // Recoverable: model can be loaded
    /// let err = ContextGraphError::Embedding(EmbeddingError::ModelNotLoaded(Embedder::Semantic));
    /// assert!(err.is_recoverable());
    ///
    /// // Not recoverable: data corruption
    /// let err = ContextGraphError::Storage(StorageError::Corruption("bad data".to_string()));
    /// assert!(!err.is_recoverable());
    /// ```
    #[inline]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Embedding(EmbeddingError::ModelNotLoaded(_))
                | Self::Storage(StorageError::Transaction(_))
                | Self::Index(IndexError::Timeout(_))
                | Self::Mcp(McpError::RateLimited(_))
                | Self::Gpu(GpuError::OutOfMemory { .. })
        )
    }

    /// Check if this error indicates a critical system issue.
    ///
    /// Critical errors indicate system health problems that require
    /// immediate attention and should be logged at ERROR level.
    ///
    /// # Returns
    ///
    /// `true` if this is a critical error, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::error::{ContextGraphError, StorageError, IndexError, GpuError};
    /// use context_graph_core::Embedder;
    ///
    /// // Critical: data corruption
    /// let err = ContextGraphError::Storage(StorageError::Corruption("bad".to_string()));
    /// assert!(err.is_critical());
    ///
    /// // Critical: index corruption
    /// let err = ContextGraphError::Index(IndexError::Corruption(
    ///     Embedder::Semantic,
    ///     "checksum mismatch".to_string()
    /// ));
    /// assert!(err.is_critical());
    ///
    /// // Not critical: validation error
    /// let err = ContextGraphError::Validation("bad input".to_string());
    /// assert!(!err.is_critical());
    /// ```
    #[inline]
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::Storage(StorageError::Corruption(_))
                | Self::Index(IndexError::Corruption(_, _))
                | Self::Gpu(GpuError::NotAvailable)
                | Self::Internal(_)
        )
    }

    /// Create an internal error from a message.
    ///
    /// Convenience method for creating internal errors.
    #[inline]
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Create a validation error from a message.
    ///
    /// Convenience method for creating validation errors.
    #[inline]
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }
}

// ============================================================================
// RESULT TYPE ALIAS
// ============================================================================

/// Result type alias for context-graph operations.
///
/// # Examples
///
/// ```rust
/// use context_graph_core::error::{Result, ContextGraphError};
///
/// fn example_operation() -> Result<String> {
///     Ok("success".to_string())
/// }
///
/// fn failing_operation() -> Result<String> {
///     Err(ContextGraphError::validation("invalid input"))
/// }
/// ```
pub type Result<T> = std::result::Result<T, ContextGraphError>;
