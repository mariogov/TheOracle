//! TeleologicalFingerprint: Complete node representation with semantic metadata.
//!
//! This is the top-level fingerprint type that wraps SemanticFingerprint with
//! content hashing, timestamps, and access tracking.

mod core;
mod types;

#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod tests;

// Re-export the main type for backwards compatibility
pub use types::TeleologicalFingerprint;
