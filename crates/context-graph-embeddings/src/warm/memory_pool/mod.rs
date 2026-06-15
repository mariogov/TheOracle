//! Memory pool management for warm model loading.
//!
//! # Overview
//!
//! This module provides isolated memory pools for the warm model loading system,
//! implementing a dual-pool architecture that separates model weights from
//! working memory allocations.
//!
//! # Pool Isolation Strategy
//!
//! The warm loading system uses two distinct memory pools with different
//! eviction policies:
//!
//! 1. **Model Pool (Non-Evictable)**: Stores model weights that must remain
//!    resident in VRAM for the entire application lifetime. These allocations
//!    are protected from memory pressure and CANNOT be evicted.
//!
//! 2. **Working Pool (Evictable)**: Stores temporary inference activations
//!    and intermediate tensors. These allocations CAN be reclaimed when
//!    memory pressure is detected.
//!
//! # Non-Evictable vs Evictable Semantics
//!
//! ## Non-Evictable (Model Pool)
//! - Allocations are permanent until explicitly freed
//! - Protected from CUDA memory pressure callbacks
//! - Failure to allocate is a fatal startup error (REQ-WARM-004)
//! - Must fit within the configured `vram_budget_bytes`
//!
//! ## Evictable (Working Pool)
//! - Allocations can be reclaimed under memory pressure
//! - Used for inference activations and temporary tensors
//! - Exhaustion returns `WorkingMemoryExhausted` (non-fatal)
//! - Sized by `vram_headroom_bytes` configuration
//!
//! # Thread-Safety Considerations
//!
//! The pools are designed to be used with `Arc<Mutex<WarmMemoryPools>>` for
//! thread-safe access from multiple inference workers. The internal state
//! is NOT internally synchronized; callers must provide external locking.
//!
//! Typical usage pattern:
//! ```rust,ignore
//! use std::sync::{Arc, Mutex};
//! use context_graph_embeddings::warm::memory_pool::WarmMemoryPools;
//!
//! let pools = Arc::new(Mutex::new(WarmMemoryPools::rtx_5090()));
//!
//! // Thread 1: Load model
//! {
//!     let mut guard = pools.lock().unwrap();
//!     guard.allocate_model("E1_Semantic", 800_000_000, vram_ptr)?;
//! }
//!
//! // Thread 2: Allocate working memory for inference
//! {
//!     let mut guard = pools.lock().unwrap();
//!     guard.allocate_working(50_000_000)?;
//! }
//! ```
//!
//! # Requirements Implemented
//!
//! - REQ-WARM-004: Non-evictable allocations for model weights
//! - REQ-WARM-005: Protected from memory pressure
//! - REQ-WARM-012: VRAM budget enforcement

mod model_pool;
mod pools;
mod types;
mod working_pool;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use model_pool::ModelMemoryPool;
pub use pools::WarmMemoryPools;
pub use types::ModelAllocation;
pub use working_pool::WorkingMemoryPool;
