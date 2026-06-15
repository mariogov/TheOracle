//! Code embedding model using Qodo/Qodo-Embed-1-1.5B.
//!
//! This model produces 1536D native vectors optimized for code understanding.
//! Based on Qwen2 architecture with Grouped-Query Attention and RoPE.
//!
//! # Dimension
//!
//! - Native output: 1536D
//! - Projected output: 1536D (no projection needed)
//!
//! # Architecture (Qwen2)
//!
//! - 28 decoder layers
//! - 1536 hidden dimension
//! - 12 attention heads with GQA (2 KV heads per layer)
//! - SwiGLU activation in FFN (8960 intermediate size)
//! - RoPE position encoding (theta=1,000,000)
//! - Last-token pooling for embedding output
//!
//! # Thread Safety
//! - `AtomicBool` for `loaded` state (lock-free reads)
//! - Inner model/tokenizer require explicit synchronization if mutable
//!
//! # Memory Layout
//! - Total estimated: ~6GB for FP32 weights (1.5B parameters)
//! - With FP16: ~3GB VRAM usage

mod attention;
mod config;
mod constants;
mod forward;
mod layers;
mod model;
mod position;
mod weights;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_batch;
#[cfg(test)]
mod tests_edge_cases;

// Re-export public types
pub use constants::{
    CODE_LATENCY_BUDGET_MS, CODE_MAX_TOKENS, CODE_MODEL_NAME, CODE_NATIVE_DIMENSION,
    CODE_PROJECTED_DIMENSION,
};
pub use model::CodeModel;
