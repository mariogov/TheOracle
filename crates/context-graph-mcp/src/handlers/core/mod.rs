//! Core Handlers struct and dispatch logic.
//!
//! PRD v6 Section 10 - Only 6 MCP tools are supported:
//! - inject_context, store_memory, get_memetic_status, search_graph
//! - trigger_consolidation
//! - merge_concepts

mod dispatch;
mod handlers;

pub use self::handlers::Handlers;
