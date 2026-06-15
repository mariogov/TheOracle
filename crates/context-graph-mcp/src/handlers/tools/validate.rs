//! Validation traits for MCP tool request DTOs.
//!
//! Provides a uniform interface for request validation, used by the
//! `parse_request` and `parse_request_validated` helpers in `helpers.rs`
//! to eliminate parse+validate boilerplate across all handler files.
//!
//! ## Two Validation Patterns
//!
//! - [`Validate`]: For DTOs whose `validate()` returns `Result<(), String>`.
//!   Used by search_causes, search_code, search_by_keywords, etc.
//!
//! - [`ValidateInto`]: For DTOs whose `validate()` parses and returns a value
//!   (e.g., `Result<Uuid, String>` or `Result<(Uuid, Uuid), String>`).
//!   Used by get_causal_chain, forget_concept, discover_graph_relationships, etc.

/// Marker trait for DTOs whose `validate()` returns `Result<(), String>`.
///
/// DTOs implementing this trait are used with `Handlers::parse_request`.
pub(crate) trait Validate {
    fn validate(&self) -> Result<(), String>;
}

/// Trait for DTOs whose `validate()` parses and returns a typed output.
///
/// DTOs implementing this trait are used with `Handlers::parse_request_validated`.
/// The `Output` type is whatever the validation produces (e.g., `Uuid`, `Vec<Uuid>`,
/// `(Uuid, Uuid)`).
pub(crate) trait ValidateInto {
    type Output;
    fn validate(&self) -> Result<Self::Output, String>;
}
