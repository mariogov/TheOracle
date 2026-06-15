//! Legacy error types retained for backwards compatibility.
//!
//! New code should use [`ContextGraphError`] instead.

use thiserror::Error;
use uuid::Uuid;

// ============================================================================
// LEGACY CORE ERROR (RETAINED FOR COMPATIBILITY)
// ============================================================================

/// Legacy error type for context-graph-core operations.
///
/// # Deprecation Notice
///
/// This type is retained for backwards compatibility. New code should use
/// [`super::ContextGraphError`] instead.
///
/// # Examples
///
/// ```rust
/// use context_graph_core::CoreError;
/// use uuid::Uuid;
///
/// fn lookup_node(id: Uuid) -> Result<(), CoreError> {
///     Err(CoreError::NodeNotFound { id })
/// }
///
/// let result = lookup_node(Uuid::nil());
/// assert!(result.is_err());
/// ```
#[derive(Debug, Error)]
pub enum CoreError {
    /// A requested node was not found in the graph.
    #[error("Node not found: {id}")]
    NodeNotFound { id: Uuid },

    /// Embedding vector dimension does not match expected size.
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A field value failed validation constraints.
    #[error("Validation error: {field} - {message}")]
    ValidationError { field: String, message: String },

    /// An error occurred during storage operations.
    #[error("Storage error: {0}")]
    StorageError(String),

    /// An error occurred with index operations.
    #[error("Index error: {0}")]
    IndexError(String),

    /// Configuration is invalid or missing.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Error during UTL computation.
    #[error("UTL computation error: {0}")]
    UtlError(String),

    /// Error during layer processing.
    #[error("Layer processing error in {layer}: {message}")]
    LayerError { layer: String, message: String },

    /// Requested feature is disabled.
    #[error("Feature disabled: {feature}")]
    FeatureDisabled { feature: String },

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Embedding error.
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Missing required field.
    #[error("Missing required field '{field}': {context}")]
    MissingField { field: String, context: String },

    /// Not implemented.
    #[error("Not implemented: {0}. See documentation for implementation guide.")]
    NotImplemented(String),

    /// Legacy format rejected.
    #[error("Legacy format rejected: {0}. See documentation for migration guide.")]
    LegacyFormatRejected(String),
}

impl From<serde_json::Error> for CoreError {
    fn from(err: serde_json::Error) -> Self {
        CoreError::SerializationError(err.to_string())
    }
}

impl From<config::ConfigError> for CoreError {
    fn from(err: config::ConfigError) -> Self {
        CoreError::ConfigError(err.to_string())
    }
}

/// Result type alias for core operations (legacy).
pub type CoreResult<T> = std::result::Result<T, CoreError>;
