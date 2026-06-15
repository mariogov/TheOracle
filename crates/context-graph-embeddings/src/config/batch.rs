//! Batch processing configuration.
//!
//! Controls how embedding requests are batched for efficient GPU utilization.

use serde::{Deserialize, Serialize};

use crate::error::{EmbeddingError, EmbeddingResult};

// ============================================================================
// PADDING STRATEGY ENUM
// ============================================================================

/// Padding strategy for variable-length sequences in a batch.
///
/// Controls how inputs of different lengths are padded to form uniform batches.
/// Choice affects memory usage and computational efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PaddingStrategy {
    /// Pad all sequences to the model's max_tokens limit.
    /// Most memory-intensive but safest for models with fixed expectations.
    MaxLength,

    /// Pad to the longest sequence in the current batch.
    /// Most memory-efficient for variable-length inputs.
    #[default]
    DynamicMax,

    /// Pad to next power of two (cache-friendly).
    /// Good for GPU memory alignment and tensor core efficiency.
    PowerOfTwo,

    /// Use predefined length buckets (64, 128, 256, 512).
    /// Balances padding efficiency with kernel optimization.
    Bucket,
}

impl PaddingStrategy {
    /// Returns all valid padding strategies.
    pub fn all() -> &'static [PaddingStrategy] {
        &[
            PaddingStrategy::MaxLength,
            PaddingStrategy::DynamicMax,
            PaddingStrategy::PowerOfTwo,
            PaddingStrategy::Bucket,
        ]
    }

    /// Returns the strategy name as snake_case string.
    pub fn as_str(&self) -> &'static str {
        match self {
            PaddingStrategy::MaxLength => "max_length",
            PaddingStrategy::DynamicMax => "dynamic_max",
            PaddingStrategy::PowerOfTwo => "power_of_two",
            PaddingStrategy::Bucket => "bucket",
        }
    }
}

// ============================================================================
// DEFAULT FUNCTIONS
// ============================================================================

fn default_max_batch_size() -> usize {
    32
}

fn default_min_batch_size() -> usize {
    1
}

fn default_max_wait_ms() -> u64 {
    50
}

fn default_dynamic_batching() -> bool {
    true
}

fn default_sort_by_length() -> bool {
    true
}

// ============================================================================
// BATCH CONFIG
// ============================================================================

/// Configuration for batch processing.
///
/// Controls how embedding requests are batched for efficient GPU utilization.
/// The batch processor accumulates requests and triggers batch inference when:
/// - Batch reaches `max_batch_size`, OR
/// - `max_wait_ms` timeout expires (if `min_batch_size` is met)
///
/// This enables high throughput (>100 items/sec) by amortizing model invocation overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum number of inputs per batch before triggering inference.
    /// Larger batches improve throughput but use more GPU memory.
    /// Constitution spec: max 32
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Minimum batch size to wait for before processing.
    /// If timeout expires and batch size >= min_batch_size, process immediately.
    /// Set to 1 for latency-sensitive applications.
    /// Default: 1
    #[serde(default = "default_min_batch_size")]
    pub min_batch_size: usize,

    /// Maximum time to wait for a full batch (milliseconds).
    /// After this time, partial batch is processed (if >= min_batch_size).
    /// Constitution spec: 50ms (latency-sensitive: 10-100ms range)
    #[serde(default = "default_max_wait_ms")]
    pub max_wait_ms: u64,

    /// Whether to enable dynamic batching based on system load.
    /// When enabled, batch sizes adjust based on queue depth and GPU utilization.
    /// Default: true
    #[serde(default = "default_dynamic_batching")]
    pub dynamic_batching: bool,

    /// Padding strategy for variable-length inputs.
    /// Controls how sequences of different lengths are padded in a batch.
    #[serde(default)]
    pub padding_strategy: PaddingStrategy,

    /// Whether to sort inputs by sequence length before batching.
    /// Reduces padding waste by grouping similar-length sequences.
    /// Can reduce padding overhead by 20-40%.
    /// Default: true
    #[serde(default = "default_sort_by_length")]
    pub sort_by_length: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: default_max_batch_size(),
            min_batch_size: default_min_batch_size(),
            max_wait_ms: default_max_wait_ms(),
            dynamic_batching: default_dynamic_batching(),
            padding_strategy: PaddingStrategy::default(),
            sort_by_length: default_sort_by_length(),
        }
    }
}

impl BatchConfig {
    /// Validate batch configuration values.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if max_batch_size is 0
    /// - `EmbeddingError::ConfigError` if min_batch_size > max_batch_size
    /// - `EmbeddingError::ConfigError` if max_wait_ms is 0 when min_batch_size > 1
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.max_batch_size == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "max_batch_size must be > 0".to_string(),
            });
        }

        if self.min_batch_size > self.max_batch_size {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "min_batch_size ({}) cannot exceed max_batch_size ({})",
                    self.min_batch_size, self.max_batch_size
                ),
            });
        }

        if self.max_wait_ms == 0 && self.min_batch_size > 1 {
            return Err(EmbeddingError::ConfigError {
                message: "max_wait_ms must be > 0 when min_batch_size > 1".to_string(),
            });
        }

        Ok(())
    }
}
