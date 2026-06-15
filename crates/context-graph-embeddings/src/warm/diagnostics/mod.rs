//! Diagnostic Dump Module for Warm Model Loading
//!
//! Provides comprehensive diagnostic reporting for the warm model loading system.
//! Supports JSON serialization for automated monitoring and human-readable output
//! for debugging.
//!
//! # Overview
//!
//! The [`WarmDiagnostics`] service generates detailed diagnostic reports that capture:
//! - System information (hostname, OS, uptime)
//! - GPU information (device, VRAM, compute capability)
//! - Memory pool status (model pool, working pool)
//! - Per-model loading state and VRAM allocations
//! - Any errors encountered during loading
//!
//! # Design Principles
//!
//! - **NO WORKAROUNDS OR FALLBACKS**: Diagnostics must be accurate
//! - **COMPREHENSIVE LOGGING**: Full context on any issue
//! - **SERIALIZABLE**: Support JSON output for automated monitoring
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::warm::{WarmLoader, WarmConfig};
//! use context_graph_embeddings::warm::diagnostics::WarmDiagnostics;
//!
//! let config = WarmConfig::default();
//! let loader = WarmLoader::new(config)?;
//!
//! // Generate and print diagnostic report
//! let report = WarmDiagnostics::generate_report(&loader);
//! println!("{}", WarmDiagnostics::to_json(&loader)?);
//!
//! // On fatal error, dump to stderr
//! WarmDiagnostics::dump_to_stderr(&loader);
//! ```

mod helpers;
mod output;
mod service;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::helpers::format_bytes;
pub use self::service::WarmDiagnostics;
pub use self::types::{
    ErrorDiagnostic, GpuDiagnostics, MemoryDiagnostics, ModelDiagnostic, SystemInfo,
    WarmDiagnosticReport,
};
