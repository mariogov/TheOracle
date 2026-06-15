//! Temporal-Periodic model struct and constructors.
//!
//! The model encodes cyclical time patterns using Fourier basis functions.

use std::sync::atomic::AtomicBool;

use crate::error::{EmbeddingError, EmbeddingResult};

use super::constants::{DEFAULT_PERIODS, HARMONICS_PER_PERIOD, TEMPORAL_PERIODIC_DIMENSION};

/// Temporal-Periodic embedding model (E3).
///
/// Encodes cyclical time patterns using Fourier basis functions.
/// This is a custom model with no pretrained weights - it computes embeddings
/// from timestamps using sin/cos of phase angles.
///
/// # Algorithm
///
/// For each period P (hour, day, week, month, year):
///   - Convert timestamp to seconds since Unix epoch
///   - Compute phase: theta = 2pi * (timestamp_secs mod P) / P
///   - Generate harmonics: sin(n*theta), cos(n*theta) for n = 1..51
///
/// This captures:
///   - Hour-of-day patterns (morning vs evening)
///   - Day-of-week patterns (weekday vs weekend)
///   - Month-of-year patterns (seasonal effects)
///
/// # Construction
///
/// ```rust,no_run
/// use context_graph_embeddings::models::TemporalPeriodicModel;
/// use context_graph_embeddings::error::EmbeddingResult;
///
/// fn example() -> EmbeddingResult<()> {
///     // Default periods (hour, day, week, month, year)
///     let model = TemporalPeriodicModel::new();
///     assert_eq!(model.periods.len(), 5);
///
///     // Custom periods (exactly 5 required for 512D output)
///     let model = TemporalPeriodicModel::with_periods(vec![
///         3600,    // Hour
///         86400,   // Day
///         604800,  // Week
///         2592000, // Month (~30 days)
///         31536000, // Year
///     ])?;
///     Ok(())
/// }
/// ```
pub struct TemporalPeriodicModel {
    /// Periods in seconds for Fourier encoding.
    pub periods: Vec<u64>,

    /// Number of harmonics per period.
    pub(crate) num_harmonics: usize,

    /// Always true for custom models (no weights to load).
    pub(crate) initialized: AtomicBool,
}

impl TemporalPeriodicModel {
    /// Create a new TemporalPeriodicModel with default periods.
    ///
    /// Uses 5 periods: hour, day, week, month, year.
    /// Model is immediately ready for use (no loading required).
    #[must_use]
    pub fn new() -> Self {
        Self {
            periods: DEFAULT_PERIODS.to_vec(),
            num_harmonics: HARMONICS_PER_PERIOD,
            initialized: AtomicBool::new(true),
        }
    }

    /// Create a model with custom periods.
    ///
    /// # Arguments
    /// * `periods` - Vector of periods in seconds.
    ///   Must have exactly 5 elements for 512D output.
    ///
    /// # Errors
    /// Returns `EmbeddingError::ConfigError` if periods doesn't have exactly 5 elements
    /// or if any period is zero.
    pub fn with_periods(periods: Vec<u64>) -> EmbeddingResult<Self> {
        if periods.len() != 5 {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "TemporalPeriodicModel requires exactly 5 periods for 512D output, got {}",
                    periods.len()
                ),
            });
        }

        // Validate periods are positive
        for (i, &period) in periods.iter().enumerate() {
            if period == 0 {
                return Err(EmbeddingError::ConfigError {
                    message: format!(
                        "Period at index {} must be positive (non-zero), got {}",
                        i, period
                    ),
                });
            }
        }

        Ok(Self {
            periods,
            num_harmonics: HARMONICS_PER_PERIOD,
            initialized: AtomicBool::new(true),
        })
    }

    /// Create a model with custom periods and harmonics.
    ///
    /// # Arguments
    /// * `periods` - Vector of periods in seconds.
    /// * `num_harmonics` - Number of frequency harmonics per period.
    ///
    /// # Errors
    /// Returns `EmbeddingError::ConfigError` if configuration is invalid.
    ///
    /// # Note
    /// The resulting dimension will be: periods.len() * num_harmonics * 2 + padding.
    /// For 512D output, use 5 periods * 51 harmonics.
    pub fn with_custom_config(periods: Vec<u64>, num_harmonics: usize) -> EmbeddingResult<Self> {
        if periods.is_empty() {
            return Err(EmbeddingError::ConfigError {
                message: "TemporalPeriodicModel requires at least one period".to_string(),
            });
        }

        if num_harmonics == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "TemporalPeriodicModel requires at least one harmonic".to_string(),
            });
        }

        // Validate periods are positive
        for (i, &period) in periods.iter().enumerate() {
            if period == 0 {
                return Err(EmbeddingError::ConfigError {
                    message: format!(
                        "Period at index {} must be positive (non-zero), got {}",
                        i, period
                    ),
                });
            }
        }

        // Check dimension matches expected
        let expected_features = periods.len() * num_harmonics * 2;
        if expected_features > TEMPORAL_PERIODIC_DIMENSION {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "Configuration produces {} features, exceeds max dimension {}",
                    expected_features, TEMPORAL_PERIODIC_DIMENSION
                ),
            });
        }

        Ok(Self {
            periods,
            num_harmonics,
            initialized: AtomicBool::new(true),
        })
    }
}

impl Default for TemporalPeriodicModel {
    fn default() -> Self {
        Self::new()
    }
}

// TemporalPeriodicModel is auto-Send+Sync: all fields (Vec<u64>, usize, AtomicBool)
// are Send+Sync. No unsafe impl needed.
