//! Tests for the Temporal-Positional embedding model (E4).
//!
//! Test modules:
//! - `construction` - Model construction and configuration tests
//! - `embedding` - Core embedding and trait implementation tests
//! - `uniqueness` - Uniqueness and determinism tests
//! - `hybrid` - Hybrid session+position encoding mode tests

mod construction;
mod embedding;
mod hybrid;
mod uniqueness;
