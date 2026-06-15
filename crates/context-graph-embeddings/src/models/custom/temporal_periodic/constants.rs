//! Constants for the Temporal-Periodic embedding model.
//!
//! Defines period durations, dimension constants, and harmonic parameters.

/// Native dimension for TemporalPeriodic model (E3).
pub const TEMPORAL_PERIODIC_DIMENSION: usize = 512;

/// Standard periods for temporal encoding (in seconds).
pub mod periods {
    /// 1 hour = 3600 seconds
    pub const HOUR: u64 = 3600;
    /// 1 day = 86400 seconds
    pub const DAY: u64 = 86400;
    /// 1 week = 604800 seconds
    pub const WEEK: u64 = 604800;
    /// 1 month = 2592000 seconds (~30 days)
    pub const MONTH: u64 = 2592000;
    /// 1 year = 31536000 seconds (~365 days)
    pub const YEAR: u64 = 31536000;
}

/// Default periods for encoding: hour, day, week, month, year.
pub const DEFAULT_PERIODS: [u64; 5] = [
    periods::HOUR,
    periods::DAY,
    periods::WEEK,
    periods::MONTH,
    periods::YEAR,
];

/// Number of frequency harmonics per period (5 periods × 51 harmonics × 2 = 510, pad to 512).
pub const HARMONICS_PER_PERIOD: usize = 51;

/// Features per period including sin and cos (51 harmonics × 2 = 102 per period).
/// Used in tests to verify periodicity of specific feature blocks.
pub const FEATURES_PER_PERIOD: usize = 102;
