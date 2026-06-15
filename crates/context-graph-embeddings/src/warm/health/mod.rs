//! Health Check API for Warm Model Loading
//!
//! Provides real-time health status reporting for the warm loading system.
//! This module is critical for monitoring, orchestration, and debugging.
//!
//! # Design Principles
//!
//! - **NO WORKAROUNDS OR FALLBACKS**: Health checks report the true system state
//! - **NO MOCK DATA**: All status information comes from actual components
//! - **THREAD SAFE**: Safe for concurrent access from multiple monitoring threads
//!
//! # Health States
//!
//! | Status | Condition |
//! |--------|-----------|
//! | `Healthy` | All registered models are in `Warm` state |
//! | `Loading` | At least one model is `Loading` or `Validating`, none `Failed` |
//! | `Unhealthy` | At least one model is in `Failed` state |
//! | `NotInitialized` | No models registered or registry unavailable |
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::warm::health::{WarmHealthChecker, WarmHealthStatus};
//! use context_graph_embeddings::warm::WarmLoader;
//!
//! let loader = WarmLoader::new(config)?;
//! let checker = WarmHealthChecker::from_loader(&loader);
//!
//! // Quick status check
//! if checker.is_healthy() {
//!     println!("All models ready for inference");
//! }
//!
//! // Detailed health check
//! let health = checker.check();
//! println!("Status: {:?}, Models warm: {}/{}",
//!     health.status, health.models_warm, health.models_total);
//! ```
//!
//! # Requirements Implemented
//!
//! - REQ-WARM-006: Health check status reporting
//! - REQ-WARM-007: Per-model state visibility

mod check;
mod checker;
mod status;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_additional;

// Re-export all public types for backwards compatibility
pub use check::WarmHealthCheck;
pub use checker::WarmHealthChecker;
pub use status::WarmHealthStatus;
