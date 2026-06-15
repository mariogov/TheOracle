//! TemporalRecentModel struct and construction.

use std::sync::atomic::{AtomicBool, Ordering};

use chrono::{DateTime, Utc};

use crate::error::{EmbeddingError, EmbeddingResult};

use super::constants::DEFAULT_DECAY_RATES;

/// Temporal-Recent embedding model (E2).
///
/// Encodes temporal recency using exponential decay across multiple time scales.
/// This is a custom model with no pretrained weights - it computes embeddings
/// from timestamps using pure mathematical operations.
///
/// # Construction
///
/// ```rust,no_run
/// use context_graph_embeddings::models::TemporalRecentModel;
/// use chrono::Utc;
///
/// // Default decay rates
/// let model = TemporalRecentModel::new();
/// assert!(model.is_initialized());
///
/// // Custom decay rates (exactly 4 required for 512D output)
/// let model = TemporalRecentModel::with_decay_rates(vec![
///     1.0 / 3600.0,   // Hour scale
///     1.0 / 86400.0,  // Day scale
///     1.0 / 604800.0, // Week scale
///     1.0 / 2592000.0, // Month scale
/// ]).expect("Valid decay rates");
///
/// // With fixed reference time (for reproducible tests)
/// let reference_time = Utc::now();
/// let model = TemporalRecentModel::with_reference_time(reference_time);
/// assert_eq!(model.reference_time(), Some(reference_time));
/// ```
pub struct TemporalRecentModel {
    /// Decay rates for different time scales (reciprocal seconds).
    pub(crate) decay_rates: Vec<f32>,

    /// Reference time for relative calculations (None = use current time).
    pub(crate) reference_time: Option<DateTime<Utc>>,

    /// Always true for custom models (no weights to load).
    pub(crate) initialized: AtomicBool,
}

impl TemporalRecentModel {
    /// Create a new TemporalRecentModel with default decay rates.
    ///
    /// Uses 4 time scales: hour, day, week, month.
    /// Model is immediately ready for use (no loading required).
    #[must_use]
    pub fn new() -> Self {
        Self {
            decay_rates: DEFAULT_DECAY_RATES.to_vec(),
            reference_time: None,
            initialized: AtomicBool::new(true),
        }
    }

    /// Create a model with custom decay rates.
    ///
    /// # Arguments
    /// * `decay_rates` - Vector of decay rates (reciprocal seconds).
    ///   Must have exactly 4 elements for 512D output.
    ///
    /// # Errors
    /// Returns `EmbeddingError::ConfigError` if decay_rates doesn't have exactly 4 elements.
    pub fn with_decay_rates(decay_rates: Vec<f32>) -> EmbeddingResult<Self> {
        if decay_rates.len() != 4 {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "TemporalRecentModel requires exactly 4 decay rates for 512D output, got {}",
                    decay_rates.len()
                ),
            });
        }

        // Validate decay rates are positive
        for (i, &rate) in decay_rates.iter().enumerate() {
            if rate <= 0.0 || !rate.is_finite() {
                return Err(EmbeddingError::ConfigError {
                    message: format!(
                        "Decay rate at index {} must be positive and finite, got {}",
                        i, rate
                    ),
                });
            }
        }

        Ok(Self {
            decay_rates,
            reference_time: None,
            initialized: AtomicBool::new(true),
        })
    }

    /// Create a model with a fixed reference time.
    ///
    /// Useful for reproducible tests where you want consistent output
    /// regardless of when the test runs.
    ///
    /// # Arguments
    /// * `reference_time` - Fixed reference point for time delta calculations.
    #[must_use]
    pub fn with_reference_time(reference_time: DateTime<Utc>) -> Self {
        Self {
            decay_rates: DEFAULT_DECAY_RATES.to_vec(),
            reference_time: Some(reference_time),
            initialized: AtomicBool::new(true),
        }
    }

    /// Check if the model is initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Get the decay rates.
    #[inline]
    pub fn decay_rates(&self) -> &[f32] {
        &self.decay_rates
    }

    /// Get the reference time.
    #[inline]
    pub fn reference_time(&self) -> Option<DateTime<Utc>> {
        self.reference_time
    }
}

impl Default for TemporalRecentModel {
    fn default() -> Self {
        Self::new()
    }
}

// TemporalRecentModel is auto-Send+Sync: all fields (Vec<f32>, Option<DateTime<Utc>>,
// AtomicBool) are Send+Sync. No unsafe impl needed.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_initialized_model() {
        let model = TemporalRecentModel::new();

        assert!(
            model.is_initialized(),
            "Custom model must be initialized immediately"
        );
        assert_eq!(model.decay_rates.len(), 4, "Must have 4 decay rates");
    }

    #[test]
    fn test_default_decay_rates() {
        let model = TemporalRecentModel::new();

        assert_eq!(model.decay_rates[0], 1.0 / 3600.0, "Hour scale");
        assert_eq!(model.decay_rates[1], 1.0 / 86400.0, "Day scale");
        assert_eq!(model.decay_rates[2], 1.0 / 604800.0, "Week scale");
        assert_eq!(model.decay_rates[3], 1.0 / 2592000.0, "Month scale");
    }

    #[test]
    fn test_custom_decay_rates_valid() {
        let rates = vec![0.1, 0.01, 0.001, 0.0001];
        let model = TemporalRecentModel::with_decay_rates(rates.clone()).expect("Should succeed");

        assert_eq!(model.decay_rates, rates);
    }

    #[test]
    fn test_custom_decay_rates_wrong_count() {
        let rates = vec![0.1, 0.01, 0.001]; // Only 3

        let result = TemporalRecentModel::with_decay_rates(rates);

        assert!(result.is_err(), "Should fail with wrong count");
        match result {
            Err(EmbeddingError::ConfigError { message }) => {
                assert!(
                    message.contains("exactly 4"),
                    "Error should mention 4 rates"
                );
            }
            Err(other) => panic!("Expected ConfigError, got {:?}", other),
            Ok(_) => panic!("Expected ConfigError, got Ok"),
        }
    }

    #[test]
    fn test_custom_decay_rates_zero_invalid() {
        let rates = vec![0.1, 0.0, 0.001, 0.0001]; // Zero rate

        let result = TemporalRecentModel::with_decay_rates(rates);

        assert!(result.is_err(), "Zero rate should fail");
    }

    #[test]
    fn test_custom_decay_rates_negative_invalid() {
        let rates = vec![0.1, -0.01, 0.001, 0.0001]; // Negative rate

        let result = TemporalRecentModel::with_decay_rates(rates);

        assert!(result.is_err(), "Negative rate should fail");
    }

    #[test]
    fn test_with_reference_time() {
        let ref_time = Utc::now();
        let model = TemporalRecentModel::with_reference_time(ref_time);

        assert_eq!(model.reference_time, Some(ref_time));
        assert!(model.is_initialized());
    }

    #[test]
    fn test_default_impl() {
        let model = TemporalRecentModel::default();

        assert!(model.is_initialized());
        assert_eq!(model.decay_rates.len(), 4);
    }

    #[test]
    fn test_model_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TemporalRecentModel>();
    }

    #[test]
    fn test_model_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TemporalRecentModel>();
    }
}
