//! Model validation logic for warm-loaded models.
//!
//! This module validates that models loaded into VRAM are correct before
//! marking them as Warm. Validation includes:
//!
//! - **Dimension validation**: Output dimensions match expected values
//! - **Weight validation**: No NaN or Inf values in model weights
//! - **Checksum validation**: Weight checksums match expected values
//! - **Inference validation**: Test inference produces valid output
//!
//! # Requirements
//!
//! - REQ-WARM-011: Model dimension validation
//! - REQ-WARM-003: Cold-start validation (test inference)
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Validation failures return errors, never silently pass
//! - **FAST FAIL**: Validation runs immediately after model load
//! - **COMPREHENSIVE**: Multiple validation stages catch different failure modes

mod comparisons;
mod config;
mod result;
mod validator;
mod validator_model;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::config::{TestInferenceConfig, TestInput};
pub use self::result::ValidationResult;
pub use self::validator::WarmValidator;
