//! LoadedModelWeights - Complete set of weights for a model loaded into GPU memory.
//!
//! # Constitution Alignment
//!
//! - **REQ-WARM-003**: Non-evictable VRAM allocation (tensors are pinned)
//! - **REQ-WARM-005**: Weight integrity verification (file_checksum)
//! - **AP-007**: No stub data in production (all tensors are real GPU data)

use crate::gpu::GpuTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Complete set of weights for a model loaded into GPU memory.
///
/// Represents a fully loaded model with all tensors resident in GPU VRAM.
/// This is the primary output of the warm loading pipeline.
///
/// # Constitution Alignment
///
/// - REQ-WARM-003: Non-evictable VRAM allocation (tensors are pinned)
/// - REQ-WARM-005: Weight integrity verification (file_checksum)
/// - AP-007: No stub data in production (all tensors are real GPU data)
///
/// # CRITICAL: No Simulation
///
/// This represents REAL weights loaded into REAL GPU memory.
/// All GpuTensor instances must be backed by actual CUDA allocations.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use context_graph_embeddings::gpu::GpuTensor;
/// use context_graph_embeddings::warm::loader::types::LoadedModelWeights;
///
/// // After loading real tensors to GPU:
/// let mut tensors = HashMap::new();
/// tensors.insert("embeddings.weight".to_string(), gpu_tensor_1);
/// tensors.insert("encoder.weight".to_string(), gpu_tensor_2);
///
/// let weights = LoadedModelWeights::new(
///     "E1_Semantic".to_string(),
///     tensors,
///     [0xAB; 32],           // Real SHA256
///     100_000_000,          // ~100MB GPU memory
///     0,                    // CUDA device 0
/// );
///
/// assert!(weights.has_tensor("embeddings.weight"));
/// ```
#[derive(Debug)]
pub struct LoadedModelWeights {
    /// Model identifier (e.g., "E1_Semantic", "E2_Code").
    ///
    /// # Invariant
    /// MUST be non-empty. Empty identifier = PANIC.
    pub model_id: String,

    /// Named tensors loaded to GPU.
    ///
    /// Uses existing GpuTensor from crate::gpu module.
    /// Key is the tensor name from SafeTensors (e.g., "encoder.layer.0.weight").
    ///
    /// # Invariant
    /// MUST be non-empty. Model must have at least one tensor.
    pub tensors: HashMap<String, GpuTensor>,

    /// SHA256 checksum of source weight file.
    ///
    /// # Invariant
    /// MUST be non-zero. All-zero checksum = PANIC.
    /// Used to verify weight file integrity.
    pub file_checksum: [u8; 32],

    /// Total GPU memory used (bytes).
    ///
    /// # Invariant
    /// MUST be > 0. Sum of all tensor memory allocations.
    pub total_gpu_bytes: usize,

    /// CUDA device where weights are loaded.
    ///
    /// 0 = first GPU, 1 = second GPU, etc.
    pub device_id: u32,

    /// Timestamp when weights were loaded.
    ///
    /// Used for performance monitoring and cache management.
    pub loaded_at: Instant,
}

impl LoadedModelWeights {
    /// Create new LoadedModelWeights with validation.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier (must be non-empty)
    /// * `tensors` - HashMap of named GpuTensors (must be non-empty)
    /// * `file_checksum` - SHA256 of source file (must be non-zero)
    /// * `total_gpu_bytes` - Total GPU memory used (must be > 0)
    /// * `device_id` - CUDA device index
    ///
    /// # Panics
    ///
    /// - If `model_id` is empty
    /// - If `tensors` is empty
    /// - If `file_checksum` is all zeros
    /// - If `total_gpu_bytes` is 0
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    #[must_use]
    pub fn new(
        model_id: String,
        tensors: HashMap<String, GpuTensor>,
        file_checksum: [u8; 32],
        total_gpu_bytes: usize,
        device_id: u32,
    ) -> Self {
        // FAIL-FAST: Empty model ID
        assert!(
            !model_id.is_empty(),
            "CONSTITUTION VIOLATION AP-007: model_id is empty. \
             Model must have a valid identifier."
        );

        // FAIL-FAST: No tensors
        assert!(
            !tensors.is_empty(),
            "CONSTITUTION VIOLATION AP-007: tensors is empty. \
             Model must have at least one tensor. \
             This indicates loading failure or corrupted weight file."
        );

        // FAIL-FAST: Zero checksum
        assert!(
            file_checksum != [0u8; 32],
            "CONSTITUTION VIOLATION AP-007: file_checksum is all zeros. \
             Real SHA256 checksum required for weight integrity."
        );

        // FAIL-FAST: Zero GPU bytes
        assert!(
            total_gpu_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: total_gpu_bytes is 0. \
             Loaded model must occupy GPU memory."
        );

        Self {
            model_id,
            tensors,
            file_checksum,
            total_gpu_bytes,
            device_id,
            loaded_at: Instant::now(),
        }
    }

    /// Get a specific tensor by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name (e.g., "encoder.layer.0.weight")
    ///
    /// # Returns
    ///
    /// Reference to the GpuTensor if found, None otherwise.
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    /// Check if a tensor exists by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to check
    ///
    /// # Returns
    ///
    /// `true` if tensor exists in this model's weights.
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get all tensor names.
    ///
    /// # Returns
    ///
    /// Iterator over tensor name strings.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Get the number of tensors in this model.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Verify checksum matches expected value.
    ///
    /// # Arguments
    ///
    /// * `expected` - Expected SHA256 checksum
    ///
    /// # Returns
    ///
    /// `true` if checksums match exactly.
    #[must_use]
    pub fn verify_checksum(&self, expected: &[u8; 32]) -> bool {
        self.file_checksum == *expected
    }

    /// Get time since loading completed.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.loaded_at.elapsed()
    }

    /// Get checksum as hex string for display/logging.
    #[must_use]
    pub fn checksum_hex(&self) -> String {
        self.file_checksum
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}
