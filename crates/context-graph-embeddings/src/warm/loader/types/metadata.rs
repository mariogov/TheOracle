//! Core metadata types: TensorMetadata and WarmLoadResult.
//!
//! # Constitution Alignment
//!
//! - **AP-007**: No Stub Data in Production - All fields contain REAL validated data
//! - **REQ-WARM-003**: Non-evictable VRAM allocation
//! - **REQ-WARM-005**: Weight integrity verification via SHA256 checksums

use candle_core::DType;
use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// CRITICAL: NO SIMULATION - ALL DATA MUST BE REAL
// Constitution AP-007: "No Stub Data in Production"
// =============================================================================

/// Metadata extracted from SafeTensors file header.
///
/// Contains the shape and type information for all tensors in a model weight file.
/// This metadata is parsed from the SafeTensors header before actual weight loading.
///
/// # Constitution Alignment
///
/// - AP-007: No stub data - shapes must reflect actual SafeTensors content
///
/// # CRITICAL: No Simulation
///
/// All fields must reflect actual SafeTensors header content. This struct will
/// PANIC if constructed with invalid data.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use candle_core::DType;
/// use context_graph_embeddings::warm::loader::types::TensorMetadata;
///
/// let mut shapes = HashMap::new();
/// shapes.insert("embeddings.weight".to_string(), vec![30522, 768]);
/// shapes.insert("encoder.layer.0.attention.self.query.weight".to_string(), vec![768, 768]);
///
/// let metadata = TensorMetadata::new(shapes, DType::F32, 24_030_504);
/// assert!(metadata.verify_params());
/// ```
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name -> shape mapping.
    ///
    /// Example: `{"embeddings.weight": [30522, 768], "layer.0.weight": [768, 768]}`
    ///
    /// # Invariant
    /// Must be non-empty. Empty shapes indicates corrupted or incomplete SafeTensors file.
    pub shapes: HashMap<String, Vec<usize>>,

    /// Data type of tensors (from candle_core).
    ///
    /// Common values: DType::F32, DType::F16, DType::BF16
    pub dtype: DType,

    /// Total number of parameters across all tensors.
    ///
    /// # Invariant
    /// MUST be > 0 for valid models. Zero parameters indicates empty/corrupted model.
    pub total_params: usize,
}

impl TensorMetadata {
    /// Create new TensorMetadata with validation.
    ///
    /// # Arguments
    ///
    /// * `shapes` - HashMap mapping tensor names to their shapes
    /// * `dtype` - Data type of the tensors
    /// * `total_params` - Total parameter count (should match sum of shape products)
    ///
    /// # Panics
    ///
    /// - If `shapes` is empty (no tensors in SafeTensors file)
    /// - If `total_params` is 0 (empty model)
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let shapes: HashMap<String, Vec<usize>> = [
    ///     ("layer.weight".to_string(), vec![768, 768])
    /// ].into_iter().collect();
    ///
    /// let metadata = TensorMetadata::new(shapes, DType::F32, 589_824);
    /// ```
    #[must_use]
    pub fn new(shapes: HashMap<String, Vec<usize>>, dtype: DType, total_params: usize) -> Self {
        // FAIL-FAST: Empty shapes means corrupted SafeTensors
        assert!(
            !shapes.is_empty(),
            "CONSTITUTION VIOLATION AP-007: shapes is empty. \
             SafeTensors must contain at least one tensor. \
             This indicates corrupted or incomplete weight file."
        );

        // FAIL-FAST: Zero params means empty model
        assert!(
            total_params > 0,
            "CONSTITUTION VIOLATION AP-007: total_params is 0. \
             Model must have parameters. \
             This indicates corrupted or empty weight file."
        );

        Self {
            shapes,
            dtype,
            total_params,
        }
    }

    /// Calculate total parameters from shapes (for verification).
    ///
    /// Computes the sum of all shape products to verify against stored total_params.
    ///
    /// # Returns
    ///
    /// Sum of element counts across all tensors.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // For shapes: {"a": [100, 768], "b": [768, 768]}
    /// // Returns: 100*768 + 768*768 = 76800 + 589824 = 666624
    /// let calculated = metadata.calculate_total_params();
    /// ```
    #[must_use]
    pub fn calculate_total_params(&self) -> usize {
        self.shapes
            .values()
            .map(|shape| shape.iter().product::<usize>())
            .sum()
    }

    /// Verify total_params matches calculated value.
    ///
    /// # Returns
    ///
    /// `true` if stored total_params equals the sum of all shape products.
    ///
    /// # Use Case
    ///
    /// Call this after parsing SafeTensors header to verify consistency.
    #[must_use]
    pub fn verify_params(&self) -> bool {
        self.total_params == self.calculate_total_params()
    }

    /// Get the number of tensors in this metadata.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.shapes.len()
    }

    /// Get shape for a specific tensor by name.
    #[must_use]
    pub fn get_shape(&self, name: &str) -> Option<&Vec<usize>> {
        self.shapes.get(name)
    }
}

/// Result of loading a model's weights into GPU memory.
///
/// Contains the GPU pointer, checksum, and metadata from a successful weight load operation.
/// This is the primary type returned by warm loading operations.
///
/// # Constitution Alignment
///
/// - REQ-WARM-003: Non-evictable VRAM allocation (gpu_ptr points to pinned memory)
/// - REQ-WARM-005: Weight integrity verification (checksum for validation)
/// - AP-007: No stub data in production (all fields are real, validated data)
///
/// # CRITICAL: No Simulation
///
/// All fields contain REAL data from actual loading operations.
/// This struct will PANIC if constructed with invalid data.
///
/// # Example
///
/// ```rust,ignore
/// use std::time::Duration;
/// use std::collections::HashMap;
/// use candle_core::DType;
/// use context_graph_embeddings::warm::loader::types::{WarmLoadResult, TensorMetadata};
///
/// // After real cudaMalloc and SHA256 computation:
/// let metadata = TensorMetadata::new(
///     [("weight".to_string(), vec![768, 768])].into_iter().collect(),
///     DType::F32,
///     589_824,
/// );
///
/// let result = WarmLoadResult::new(
///     0x7fff_dead_beef,      // Real GPU pointer from cudaMalloc
///     [0xAB; 32],            // Real SHA256 checksum
///     2_359_296,             // 589824 * 4 bytes
///     Duration::from_millis(150),
///     metadata,
/// );
///
/// assert!(result.verify_checksum(&[0xAB; 32]));
/// ```
#[derive(Debug)]
pub struct WarmLoadResult {
    /// Real GPU device pointer from cudaMalloc.
    ///
    /// # Invariant
    /// MUST be non-zero. Zero pointer = PANIC.
    /// This must be an actual pointer returned by CUDA memory allocation.
    pub gpu_ptr: u64,

    /// Real SHA256 checksum of the weight file.
    ///
    /// # Invariant
    /// MUST be non-zero. All-zero checksum = PANIC.
    /// This must be computed from actual file content.
    pub checksum: [u8; 32],

    /// Actual size of weights in GPU memory (bytes).
    ///
    /// # Invariant
    /// MUST be > 0. Zero size = PANIC.
    /// This is the actual cudaMalloc allocation size.
    pub size_bytes: usize,

    /// Loading duration for performance monitoring.
    ///
    /// Measures wall-clock time from start of load to completion.
    pub load_duration: Duration,

    /// Tensor metadata from SafeTensors header.
    ///
    /// Contains shapes, dtype, and total parameter count.
    pub tensor_metadata: TensorMetadata,
}

impl WarmLoadResult {
    /// Create a new WarmLoadResult with validation.
    ///
    /// # Arguments
    ///
    /// * `gpu_ptr` - Real CUDA device pointer (must be non-zero)
    /// * `checksum` - Real SHA256 checksum (must be non-zero)
    /// * `size_bytes` - Allocation size in bytes (must be > 0)
    /// * `load_duration` - Time taken to load
    /// * `tensor_metadata` - Parsed SafeTensors metadata
    ///
    /// # Panics
    ///
    /// - If `gpu_ptr` is 0 (null pointer)
    /// - If `checksum` is all zeros (invalid checksum)
    /// - If `size_bytes` is 0 (empty allocation)
    /// - If `tensor_metadata.total_params` is 0 (empty model)
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    #[must_use]
    pub fn new(
        gpu_ptr: u64,
        checksum: [u8; 32],
        size_bytes: usize,
        load_duration: Duration,
        tensor_metadata: TensorMetadata,
    ) -> Self {
        // FAIL-FAST: Null GPU pointer
        assert!(
            gpu_ptr != 0,
            "CONSTITUTION VIOLATION AP-007: gpu_ptr is null (0x0). \
             Real cudaMalloc pointer required. \
             This indicates CUDA allocation failure or simulated data."
        );

        // FAIL-FAST: Zero checksum (impossible for real SHA256)
        assert!(
            checksum != [0u8; 32],
            "CONSTITUTION VIOLATION AP-007: checksum is all zeros. \
             Real SHA256 checksum required. \
             This indicates simulated data or computation failure."
        );

        // FAIL-FAST: Zero size
        assert!(
            size_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: size_bytes is 0. \
             Real allocation size required. \
             This indicates allocation failure or simulated data."
        );

        // FAIL-FAST: Empty model (validated in TensorMetadata, but double-check)
        assert!(
            tensor_metadata.total_params > 0,
            "CONSTITUTION VIOLATION AP-007: total_params is 0. \
             Real model weights required. \
             This indicates corrupted or empty weight file."
        );

        Self {
            gpu_ptr,
            checksum,
            size_bytes,
            load_duration,
            tensor_metadata,
        }
    }

    /// Verify checksum matches expected value.
    ///
    /// # Arguments
    ///
    /// * `expected` - Expected SHA256 checksum to compare against
    ///
    /// # Returns
    ///
    /// `true` if checksums match exactly.
    #[must_use]
    pub fn verify_checksum(&self, expected: &[u8; 32]) -> bool {
        self.checksum == *expected
    }

    /// Get checksum as hex string for display/logging.
    #[must_use]
    pub fn checksum_hex(&self) -> String {
        self.checksum.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Get the bytes per parameter based on dtype.
    #[must_use]
    pub fn bytes_per_param(&self) -> usize {
        self.tensor_metadata.dtype.size_in_bytes()
    }
}
