//! Embedder name lookup by index.
//!
//! Provides a convenience function to get human-readable embedder names
//! from array indices. Delegates to `Embedder::name()` as the single source
//! of truth.

use crate::teleological::embedder::Embedder;

/// Get the canonical name for an embedder by index.
///
/// Delegates to `Embedder::name()` to ensure a single source of truth.
/// Returns "Unknown" for indices >= 13.
pub fn name(idx: usize) -> &'static str {
    Embedder::from_index(idx)
        .map(|e| e.name())
        .unwrap_or("Unknown")
}
