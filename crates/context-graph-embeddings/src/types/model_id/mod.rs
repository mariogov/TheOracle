//! ModelId enum identifying the 12 embedding models in the pipeline.
//!
//! Each variant maps to a specific model architecture with defined dimensions.
//! Custom models (Temporal*, Hdc) are implemented from scratch.
//! Pretrained models load weights from HuggingFace repositories.

mod conversions;
mod core;
mod display;
mod repository;
mod tokenizer;

#[cfg(test)]
mod tests;

// Re-export everything for backwards compatibility
pub use self::core::ModelId;
pub use self::tokenizer::TokenizerFamily;
