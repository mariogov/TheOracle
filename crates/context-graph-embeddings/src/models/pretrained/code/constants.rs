//! Constants for the Qodo-Embed code embedding model.
//!
//! These constants define the architectural parameters for the
//! Qodo/Qodo-Embed-1-1.5B model (1.5B parameter code embedding model).

/// Native dimension for Qodo-Embed embedding output.
pub const CODE_NATIVE_DIMENSION: usize = 1536;

/// Projected dimension (same as native for Qodo-Embed, no projection needed).
pub const CODE_PROJECTED_DIMENSION: usize = 1536;

/// Maximum tokens for Qodo-Embed (supports 32K context window).
pub const CODE_MAX_TOKENS: usize = 32768;

/// Latency budget in milliseconds (P95 target).
pub const CODE_LATENCY_BUDGET_MS: u32 = 10;

/// HuggingFace model repository name.
pub const CODE_MODEL_NAME: &str = "Qodo/Qodo-Embed-1-1.5B";
