//! Main search implementation for teleological matrix search.
//!
//! Contains the `TeleologicalMatrixSearch` struct and its core methods
//! for searching, clustering, and computing similarity breakdowns.

mod clustering;
mod comprehensive;
mod core;

// Re-export the main struct
pub use self::core::TeleologicalMatrixSearch;
