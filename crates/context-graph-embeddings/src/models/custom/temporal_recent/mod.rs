//! Temporal-Recent embedding model (E2) using exponential decay.
//!
//! This custom model produces 512-dimensional vectors encoding temporal recency
//! across multiple time scales using exponential decay.
//!
//! # Design
//!
//! The 512D vector is structured as:
//! - 4 time scales (hour, day, week, month) x 128 features per scale = 512 dimensions
//! - Each 128-feature block encodes decay at that scale with phase variations
//!
//! # Thread Safety
//! - `AtomicBool` for initialized state (always true for custom models)
//! - Pure computation with no shared mutable state
//!
//! # Performance
//! - Latency budget: <2ms per constitution.yaml
//! - No I/O or pretrained weights - pure CPU computation
//!
//! # Module Structure
//!
//! - `constants` - Constants and configuration values
//! - `model` - TemporalRecentModel struct and construction
//! - `compute` - Decay embedding computation logic
//! - `timestamp` - Timestamp parsing and extraction
//! - `traits` - EmbeddingModel trait implementation

mod compute;
mod constants;
mod model;
mod timestamp;
mod traits;

#[cfg(test)]
mod tests;

// Re-export public API for backwards compatibility
pub use constants::{
    DEFAULT_DECAY_RATES, FEATURES_PER_SCALE, MAX_TIME_DELTA_SECS, NUM_TIME_SCALES,
    TEMPORAL_RECENT_DIMENSION,
};
pub use model::TemporalRecentModel;

// Re-export timestamp utilities for advanced usage
pub use timestamp::{extract_timestamp, parse_timestamp};

// Re-export compute function for advanced usage
pub use compute::compute_decay_embedding;
