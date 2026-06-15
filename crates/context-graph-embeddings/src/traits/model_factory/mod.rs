//! Model factory trait and configuration types.
//!
//! This module defines the factory pattern for creating embedding model instances.
//! The factory abstracts model creation, enabling dependency injection and testability.
//!
//! # Thread Safety
//!
//! All types require `Send + Sync` for safe concurrent access.
//! Factory implementations can be shared across async tasks via `Arc<dyn ModelFactory>`.
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Invalid config returns `EmbeddingError::ConfigError`
//! - **FAIL FAST**: Unknown ModelId returns `EmbeddingError::ModelNotFound`
//! - **CONSERVATIVE ESTIMATES**: Memory estimates are overestimates, never underestimates
//!
//! # Module Structure
//!
//! - [`device`]: Device placement options (CPU, CUDA, Auto)
//! - [`quantization`]: Quantization modes (None, Int8, FP16, BF16)
//! - [`config`]: Single model configuration
//! - [`trait_def`]: The ModelFactory trait definition
//! - [`memory`]: Memory estimation constants and functions

mod config;
mod device;
mod memory;
mod quantization;
mod trait_def;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use config::SingleModelConfig;
pub use device::DevicePlacement;
pub use memory::{get_memory_estimate, MEMORY_ESTIMATES, TOTAL_MEMORY_ESTIMATE};
pub use quantization::QuantizationMode;
pub use trait_def::ModelFactory;
