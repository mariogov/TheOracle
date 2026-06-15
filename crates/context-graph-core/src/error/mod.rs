//! Error types for context-graph-core.
//!
//! This module defines the central error types used throughout the context-graph system:
//!
//! - [`ContextGraphError`]: Top-level unified error for all crate errors
//! - [`CoreError`]: Legacy error type (retained for compatibility)
//! - Sub-error types: [`EmbeddingError`], [`StorageError`], [`IndexError`],
//!   [`ConfigError`], [`GpuError`], [`McpError`]
//!
//! # Constitution Compliance
//!
//! Per constitution.yaml rust_standards/error_handling (lines 136-141):
//! - Use `thiserror` for library error types
//! - Never panic in library code; return Result
//! - Propagate errors with `?` operator
//! - Add context with `.context()` or `.with_context()`
//!
//! Per AP-14: No `.unwrap()` in library code - Use `.expect()` with context or return Result
//!
//! # Examples
//!
//! ```rust
//! use context_graph_core::error::{ContextGraphError, EmbeddingError, Result};
//! use context_graph_core::Embedder;
//!
//! fn generate_embedding(text: &str) -> Result<Vec<f32>> {
//!     if text.is_empty() {
//!         return Err(ContextGraphError::Embedding(EmbeddingError::EmptyInput));
//!     }
//!     // ... embedding logic
//!     Ok(vec![0.0; 1024])
//! }
//!
//! let result = generate_embedding("");
//! assert!(matches!(
//!     result,
//!     Err(ContextGraphError::Embedding(EmbeddingError::EmptyInput))
//! ));
//! ```

mod conversions;
mod legacy;
mod sub_errors;
mod unified;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use legacy::{CoreError, CoreResult};
pub use sub_errors::{ConfigError, EmbeddingError, GpuError, IndexError, McpError, StorageError};
pub use unified::ContextGraphError;

// Re-export Result type alias
pub use unified::Result;
