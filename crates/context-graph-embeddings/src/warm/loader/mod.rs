//! Warm Model Loader Orchestrator
//!
//! The main orchestrator for warm model loading. Coordinates the loading of all 12
//! embedding models into VRAM at startup, ensuring they remain resident
//! for the application lifetime.
//!
//! # Critical Design Decisions
//!
//! ## NO WORKAROUNDS OR FALLBACKS
//!
//! This loader implements a **fail-fast** strategy. If any model fails to load,
//! the entire startup MUST fail. There are no fallback modes, no degraded operation,
//! and no partial loading states.
//!
//! ## Exit Codes (101-110)
//!
//! On fatal errors, the loader calls `std::process::exit()` with codes from
//! [`WarmError::exit_code()`].

mod constants;
mod engine;
mod helpers;
mod operations;
mod preflight;
mod summary;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use constants::{DEFAULT_EMBEDDING_DIMENSION, GB, MODEL_SIZES};
pub use engine::WarmLoader;
pub use helpers::format_bytes;
pub use summary::LoadingSummary;

// Re-export warm loading types (TASK-EMB-006, TASK-EMB-016)
pub use types::{
    InferenceValidation, LoadedModelWeights, TensorMetadata, VramAllocationTracking, WarmLoadResult,
};

// Re-export weight loading operations (TASK-EMB-013)
pub use operations::{load_weights, verify_checksum, TensorMetadata as OperationsTensorMetadata};
