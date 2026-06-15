//! Custom embedding model implementations.
//!
//! Custom models (E2-E4, E9) are computed from scratch without pretrained weights.
//! They implement specialized mathematical encodings:
//! - TemporalRecent (E2): Exponential decay for recency
//! - TemporalPeriodic (E3): Fourier basis for periodicity
//! - TemporalPositional (E4): Sinusoidal positional encoding
//! - Hdc (E9): Hyperdimensional computing with 10K-bit binary hypervectors

mod hdc;
mod temporal_periodic;
mod temporal_positional;
mod temporal_recent;

pub use hdc::{
    HdcModel, Hypervector, DEFAULT_NGRAM_SIZE, DEFAULT_SEED, HDC_DIMENSION, HDC_PROJECTED_DIMENSION,
};
pub use temporal_periodic::{
    periods, TemporalPeriodicModel, DEFAULT_PERIODS, FEATURES_PER_PERIOD, HARMONICS_PER_PERIOD,
    TEMPORAL_PERIODIC_DIMENSION,
};
pub use temporal_positional::{
    TemporalPositionalModel, DEFAULT_BASE, TEMPORAL_POSITIONAL_DIMENSION,
};
pub use temporal_recent::{
    compute_decay_embedding, extract_timestamp, parse_timestamp, TemporalRecentModel,
    DEFAULT_DECAY_RATES, FEATURES_PER_SCALE, MAX_TIME_DELTA_SECS, NUM_TIME_SCALES,
    TEMPORAL_RECENT_DIMENSION,
};
