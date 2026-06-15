//! Memory tracking for GPU allocation.
//!
//! Tracks memory usage across loaded models to prevent OOM conditions.
//! All allocations are checked against the configured budget before proceeding.
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Allocation fails immediately if budget exceeded
//! - **FAIL FAST**: Invalid state returns error, never panics
//! - **CONSERVATIVE**: Budget check before allocation, not after
//!
//! # Module Structure
//!
//! - `core`: Core MemoryTracker implementation
//! - `tests/`: Comprehensive test suite (split into basic and advanced)

mod core;

#[cfg(test)]
mod tests;

// Re-export the main type for backwards compatibility
pub use self::core::MemoryTracker;
