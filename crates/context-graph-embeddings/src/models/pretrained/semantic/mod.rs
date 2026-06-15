//! Semantic embedding model using intfloat/e5-large-v2.
//!
//! This is the primary semantic understanding model (E1) producing 1024D dense vectors.
//! Uses instruction prefixes to distinguish between queries and passages.
//!
//! # GPU Acceleration
//!
//! When the `candle` feature is enabled, this model uses GPU-accelerated BERT inference
//! via Candle with the following pipeline:
//! 1. Tokenization with HuggingFace tokenizers
//! 2. GPU embedding lookup and position encoding
//! 3. GPU-accelerated transformer forward pass
//! 4. Mean pooling over sequence dimension
//! 5. L2 normalization on GPU
//!
//! # Thread Safety
//! - `AtomicBool` for `loaded` state (lock-free reads)
//! - Inner model/tokenizer require explicit synchronization if mutable
//!
//! # Memory Layout
//! - Total estimated: 1.3GB for FP32 weights
//! - With FP16 quantization: ~650MB

// Submodules
mod attention;
mod constants;
mod embeddings;
mod encoder;
mod ffn;
mod gpu_forward;
mod layer_norm;
mod loader;
mod model;
mod pooling;
mod tests;
mod trait_impl;
mod types;

// Re-export public API for backwards compatibility
pub use constants::{
    PASSAGE_PREFIX, QUERY_PREFIX, SEMANTIC_DIMENSION, SEMANTIC_LATENCY_BUDGET_MS,
    SEMANTIC_MAX_TOKENS,
};
pub use types::SemanticModel;
