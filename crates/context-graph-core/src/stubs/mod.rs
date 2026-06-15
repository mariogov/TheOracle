//! Stub implementations for development and testing.
//!
//! # ⚠️ TEST ONLY - DO NOT USE IN PRODUCTION ⚠️
//!
//! **CONSTITUTION AP-007 COMPLIANCE:**
//! All stub exports are gated with `#[cfg(any(test, feature = "test-utils"))]`.
//! Production code CANNOT import these stubs unless the `test-utils` feature is enabled.
//! This feature should NEVER be enabled in production builds.
//!
//! These implementations provide deterministic behavior for testing
//! and development. They are **not suitable for production use** due to
//! performance limitations (O(n) search) and lack of persistence.
//!
//! # Stubs (TEST ONLY)
//!
//! - [`StubMultiArrayProvider`]: Deterministic 13-embedding generation (TASK-F007)
//! - [`InMemoryTeleologicalStore`]: In-memory teleological storage (TASK-F008)
//!   - **O(n) search complexity** - full table scan
//!   - **No persistence** - data lost on drop
//!
//! # REMOVED (Phase 1 Cleanup)
//!
//! - `StubEmbeddingProvider`: Deprecated single-embedding stub has been DELETED.
//!   Use `StubMultiArrayProvider` for tests requiring embedding generation.
//!
//! # Usage
//!
//! These stubs are available in test code and when `test-utils` feature is enabled:
//!
//! ```ignore
//! // In Cargo.toml for downstream test crates:
//! // [dev-dependencies]
//! // context-graph-core = { workspace = true, features = ["test-utils"] }
//!
//! use context_graph_core::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider};
//!
//! fn test_example() {
//!     let store = InMemoryTeleologicalStore::new();
//!     let provider = StubMultiArrayProvider::new();
//! }
//! ```
//!
//! # Production Alternatives
//!
//! For production use, you MUST use:
//! - `RocksDbTeleologicalStore` from `context-graph-storage` (with HNSW indexing)
//! - Real GPU embedding providers from `context-graph-embeddings`
//!
//! Attempting to use stubs in production will result in a compile error.

// AP-007: All stub modules are test-only or test-utils feature
#[cfg(any(test, feature = "test-utils"))]
mod multi_array_stub;
#[cfg(any(test, feature = "test-utils"))]
mod teleological_store_stub;

// AP-007: All stub exports are gated to test-only or test-utils builds
// Production code CANNOT import these - compile error if attempted

// Multi-array embedding stub (TASK-F007) - TEST ONLY
#[cfg(any(test, feature = "test-utils"))]
pub use multi_array_stub::StubMultiArrayProvider;

// Teleological memory store stub (TASK-F008) - TEST ONLY
#[cfg(any(test, feature = "test-utils"))]
pub use teleological_store_stub::InMemoryTeleologicalStore;
