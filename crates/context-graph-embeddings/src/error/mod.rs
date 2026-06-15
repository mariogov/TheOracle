//! Comprehensive error type for all embedding pipeline failures.
//!
//! # Error Categories
//!
//! | Category | Variants | Recovery Strategy |
//! |----------|----------|-------------------|
//! | Model | ModelNotFound, ModelLoadError, NotInitialized | Retry with different config |
//! | Validation | InvalidDimension, InvalidValue, EmptyInput, InputTooLong | Fix input data |
//! | Processing | BatchError, TokenizationError | Retry or fallback model |
//! | Infrastructure | GpuError, CacheError, IoError, Timeout | Retry or degrade |
//! | Configuration | ConfigError, UnsupportedModality | Fix configuration |
//! | Serialization | SerializationError | Fix data format |
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Errors must propagate, not be silently handled
//! - **FAIL FAST**: Invalid state triggers immediate error
//! - **CONTEXTUAL**: Every variant includes debugging information
//! - **TRACEABLE**: Error chain preserved via `source`

mod types;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_extended;

// Re-export all public types
pub use types::{EmbeddingError, EmbeddingResult, ErrorSeverity};
