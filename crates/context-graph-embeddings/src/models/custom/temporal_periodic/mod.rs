//! Temporal-Periodic embedding model (E3) using Fourier basis functions.
//!
//! This custom model produces 512-dimensional vectors encoding cyclical time patterns
//! using Fourier basis functions (sin/cos pairs) at multiple periods.
//!
//! # Design
//!
//! The 512D vector is structured as:
//! - 5 periods (hour, day, week, month, year) x 102 features per period = 510 dimensions + 2 padding = 512
//! - Each 102-feature block contains 51 harmonics x 2 (sin, cos)
//!
//! # Mathematical Foundation
//!
//! For each period P:
//! - Compute phase: theta = 2pi * (timestamp_secs mod P) / P
//! - Generate harmonics: sin(n*theta), cos(n*theta) for n = 1..51
//!
//! This captures cyclical patterns:
//! - Hour-of-day (morning vs evening)
//! - Day-of-week (weekday vs weekend)
//! - Week-of-month
//! - Month-of-year (seasonal effects)
//! - Time-of-year (annual cycles)
//!
//! # Thread Safety
//! - `AtomicBool` for initialized state (always true for custom models)
//! - Pure computation with no shared mutable state
//!
//! # Performance
//! - Latency budget: <2ms per constitution.yaml
//! - No I/O or pretrained weights - pure CPU computation

mod constants;
mod embed_impl;
mod encoding;
mod model;

#[cfg(test)]
mod tests;

// Re-export all public items for backwards compatibility
pub use constants::{
    periods, DEFAULT_PERIODS, FEATURES_PER_PERIOD, HARMONICS_PER_PERIOD,
    TEMPORAL_PERIODIC_DIMENSION,
};
pub use model::TemporalPeriodicModel;
