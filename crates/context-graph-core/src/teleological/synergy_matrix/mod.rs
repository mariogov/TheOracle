//! 13x13 Cross-embedding synergy matrix for teleological fusion.
//!
//! The synergy matrix captures the strength of relationships between different
//! embedding spaces. High synergy pairs should have their cross-correlations
//! amplified in teleological fusion.
//!
//! From teleoplan.md:
//! - Diagonal is always 1.0 (self-synergy)
//! - Matrix must be symmetric
//! - Values in [0.0, 1.0]
//! - Base synergies: weak (0.3), moderate (0.6), strong (0.9)

mod accessors;
mod analysis;
mod constants;
mod constructors;
mod presets;
mod types;
mod validation;

#[cfg(test)]
mod tests;

// Re-export all public items to maintain backwards compatibility
pub use constants::{BASE_SYNERGIES, CROSS_CORRELATION_COUNT, SYNERGY_DIM};
pub use types::SynergyMatrix;
