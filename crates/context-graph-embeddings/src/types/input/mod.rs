//! Multi-modal input types for the embedding pipeline.
//!
//! ModelInput provides a unified interface for passing different types of content
//! to the embedding models, allowing each model to handle inputs it supports.
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Empty content returns `EmbeddingError::EmptyInput` immediately
//! - **NO MOCK DATA**: All validation is real, no stubs
//! - **DETERMINISTIC HASHING**: `content_hash()` uses xxhash64 for cache keying
//!
//! # Supported Input Types
//!
//! | Variant | Models | Metadata |
//! |---------|--------|----------|
//! | Text | E1, E5-E6, E8-E9, E11-E12 | Optional instruction prefix |
//! | Code | E7 (CodeBERT) | Language identifier |
//! | Image | E10 (e5-base-v2) | Image format |
//! | Audio | Future | Sample rate, channels |

mod accessors;
mod image_format;
mod input_type;
mod model_input;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use image_format::ImageFormat;
pub use input_type::InputType;
pub use model_input::ModelInput;
