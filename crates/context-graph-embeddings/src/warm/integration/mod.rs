//! Pipeline Integration Module for Warm Model Loading
//!
//! Connects [`WarmLoader`] with the embedding pipeline to provide a unified
//! warmed embedding system with all models pre-loaded in VRAM at startup.
//!
//! # Critical Design Decisions
//!
//! ## NO WORKAROUNDS OR FALLBACKS
//!
//! This integration implements a **fail-fast** strategy. If any component
//! fails during initialization, the pipeline terminates immediately with
//! an appropriate exit code. There are no:
//!
//! - Partial initialization modes
//! - Degraded operation states
//! - Mock or fallback models
//!
//! ## Exit Behavior
//!
//! On fatal errors, [`WarmEmbeddingPipeline::create_and_warm()`] calls
//! `std::process::exit()` with the error's exit code (101-110).
//!
//! # Requirements Implemented
//!
//! - REQ-WARM-001: Load all 12 embedding models at startup
//! - REQ-WARM-003: Validate models with test inference
//! - REQ-WARM-006: Health check status reporting
//! - REQ-WARM-007: Per-model state visibility
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::warm::{WarmConfig, WarmEmbeddingPipeline};
//!
//! // Production usage - fails fast on any error
//! let pipeline = WarmEmbeddingPipeline::create_and_warm(WarmConfig::default())?;
//!
//! // Check readiness
//! assert!(pipeline.is_ready());
//!
//! // Access health status
//! let health = pipeline.health();
//! println!("Status: {:?}, Models: {}/{}",
//!     health.status, health.models_warm, health.models_total);
//!
//! // Access registry for model handles
//! let registry = pipeline.registry();
//! ```

mod pipeline;

#[cfg(test)]
mod tests;

// Re-export all public items for backwards compatibility
pub use pipeline::WarmEmbeddingPipeline;
