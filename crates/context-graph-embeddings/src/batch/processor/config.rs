//! BatchProcessor configuration.
//!
//! Contains configuration types and validation for the BatchProcessor.

use crate::config::BatchConfig;
use crate::error::{EmbeddingError, EmbeddingResult};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the BatchProcessor.
#[derive(Debug, Clone)]
pub struct BatchProcessorConfig {
    /// Per-model batch configuration.
    pub batch_config: BatchConfig,

    /// How often to check queues for timeout (default: 10ms).
    pub poll_interval_ms: u64,

    /// Maximum concurrent batches across all models (default: 4).
    /// Limits GPU memory pressure.
    pub max_concurrent_batches: usize,

    /// Channel buffer size for incoming requests (default: 1000).
    pub request_buffer_size: usize,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            batch_config: BatchConfig::default(),
            poll_interval_ms: 10,
            max_concurrent_batches: 4,
            request_buffer_size: 1000,
        }
    }
}

impl BatchProcessorConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// * `EmbeddingError::ConfigError` if configuration is invalid
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.max_concurrent_batches == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "max_concurrent_batches must be > 0".to_string(),
            });
        }
        if self.request_buffer_size == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "request_buffer_size must be > 0".to_string(),
            });
        }
        if self.poll_interval_ms == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "poll_interval_ms must be > 0".to_string(),
            });
        }
        // Validate nested batch config
        self.batch_config.validate()?;
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BatchProcessorConfig::default();

        assert_eq!(config.poll_interval_ms, 10);
        assert_eq!(config.max_concurrent_batches, 4);
        assert_eq!(config.request_buffer_size, 1000);
    }

    #[test]
    fn test_config_validate_success() {
        let config = BatchProcessorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_concurrent_batches() {
        let mut config = BatchProcessorConfig::default();
        config.max_concurrent_batches = 0;

        let result = config.validate();
        assert!(result.is_err());
        if let Err(EmbeddingError::ConfigError { message }) = result {
            assert!(message.contains("max_concurrent_batches"));
        }
    }

    #[test]
    fn test_config_validate_zero_buffer_size() {
        let mut config = BatchProcessorConfig::default();
        config.request_buffer_size = 0;

        let result = config.validate();
        assert!(result.is_err());
        if let Err(EmbeddingError::ConfigError { message }) = result {
            assert!(message.contains("request_buffer_size"));
        }
    }

    #[test]
    fn test_config_validate_zero_poll_interval() {
        let mut config = BatchProcessorConfig::default();
        config.poll_interval_ms = 0;

        let result = config.validate();
        assert!(result.is_err());
        if let Err(EmbeddingError::ConfigError { message }) = result {
            assert!(message.contains("poll_interval_ms"));
        }
    }

    #[test]
    fn test_config_clone() {
        let config = BatchProcessorConfig::default();
        let cloned = config.clone();

        assert_eq!(config.poll_interval_ms, cloned.poll_interval_ms);
        assert_eq!(config.max_concurrent_batches, cloned.max_concurrent_batches);
        assert_eq!(config.request_buffer_size, cloned.request_buffer_size);
    }
}
