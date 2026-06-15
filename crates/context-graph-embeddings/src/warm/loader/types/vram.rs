//! VRAM Allocation Tracking types (TASK-EMB-016).
//!
//! # Constitution Alignment
//!
//! - AP-007: All values MUST come from real CUDA API calls
//! - REQ-WARM-003: Non-evictable VRAM allocation tracking

/// GPU VRAM allocation tracking with real/fake detection.
///
/// # Constitution Alignment
///
/// - AP-007: All values MUST come from real CUDA API calls
/// - REQ-WARM-003: Non-evictable VRAM allocation tracking
///
/// # CRITICAL: No Simulation
///
/// Fake values are FORBIDDEN. The `is_real()` method detects known fake patterns:
/// - Fake pointer `0x7f80_0000_0000`
/// - VRAM delta mismatches (claims 1GB delta for 1KB allocation)
/// - Zero-size allocations
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::warm::loader::types::VramAllocationTracking;
///
/// // From real CUDA calls:
/// let tracking = VramAllocationTracking::new(
///     0x7fff_0000_1000,  // Real cudaMalloc pointer
///     104_857_600,       // 100MB allocation
///     5000,              // 5GB VRAM before
///     5100,              // 5.1GB VRAM after (100MB delta)
/// );
///
/// assert!(tracking.is_real());
/// tracking.assert_real(); // Panics on fake data
/// ```
#[derive(Debug, Clone)]
pub struct VramAllocationTracking {
    /// Base pointer on GPU (from cudaMalloc).
    ///
    /// # Invariant
    /// MUST NOT be 0 (null) or 0x7f80_0000_0000 (known fake value).
    pub base_ptr: u64,

    /// Total bytes allocated.
    ///
    /// # Invariant
    /// MUST be > 0. Zero allocation indicates failed or fake allocation.
    pub size_bytes: usize,

    /// VRAM used before loading (from cudaMemGetInfo), in MB.
    pub vram_before_mb: u64,

    /// VRAM used after loading (from cudaMemGetInfo), in MB.
    pub vram_after_mb: u64,

    /// Actual delta: vram_after_mb - vram_before_mb.
    ///
    /// Calculated automatically in `new()`.
    pub vram_delta_mb: u64,
}

impl VramAllocationTracking {
    /// Known fake GPU pointer value used in simulations.
    ///
    /// Constitution AP-007 forbids using this value.
    pub const FAKE_POINTER: u64 = 0x7f80_0000_0000u64;

    /// Maximum allowed delta mismatch in MB (50MB tolerance for GPU overhead).
    pub const DELTA_TOLERANCE_MB: i64 = 50;

    /// Create new VramAllocationTracking with fail-fast validation.
    ///
    /// # Arguments
    ///
    /// * `base_ptr` - Real CUDA device pointer (must be non-zero)
    /// * `size_bytes` - Allocation size in bytes (must be > 0)
    /// * `vram_before_mb` - VRAM usage before allocation in MB
    /// * `vram_after_mb` - VRAM usage after allocation in MB
    ///
    /// # Panics
    ///
    /// - If `base_ptr` is 0 (null pointer)
    /// - If `size_bytes` is 0 (empty allocation)
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tracking = VramAllocationTracking::new(
    ///     0x7fff_0000_1000,  // Real pointer
    ///     104_857_600,       // 100MB
    ///     5000,              // 5GB before
    ///     5100,              // 5.1GB after
    /// );
    /// ```
    #[must_use]
    pub fn new(base_ptr: u64, size_bytes: usize, vram_before_mb: u64, vram_after_mb: u64) -> Self {
        assert!(
            base_ptr != 0,
            "CONSTITUTION VIOLATION AP-007: base_ptr is null. \
             Real cudaMalloc pointer required."
        );
        assert!(
            size_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: size_bytes is 0. \
             Real allocation size required."
        );

        let vram_delta_mb = vram_after_mb.saturating_sub(vram_before_mb);

        Self {
            base_ptr,
            size_bytes,
            vram_before_mb,
            vram_after_mb,
            vram_delta_mb,
        }
    }

    /// Check if allocation looks real (not simulated).
    ///
    /// Returns `false` if any known fake pattern is detected:
    /// - Fake pointer (0x7f80_0000_0000)
    /// - Zero-size allocation
    /// - VRAM delta doesn't match allocation size (within 50MB tolerance)
    ///
    /// # Returns
    ///
    /// `true` if allocation appears to be from real CUDA operations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Real allocation: 100MB, delta matches
    /// let real = VramAllocationTracking::new(0x7fff_0000_1000, 104_857_600, 5000, 5100);
    /// assert!(real.is_real());
    ///
    /// // Fake: 1KB allocation with 1GB delta
    /// let fake = VramAllocationTracking {
    ///     base_ptr: 0x7fff_0000_1000,
    ///     size_bytes: 1024,
    ///     vram_before_mb: 1000,
    ///     vram_after_mb: 2000,
    ///     vram_delta_mb: 1000,
    /// };
    /// assert!(!fake.is_real());
    /// ```
    #[must_use]
    pub fn is_real(&self) -> bool {
        // Fake pointer check (common simulation value)
        if self.base_ptr == Self::FAKE_POINTER {
            return false;
        }

        // Zero allocation is suspicious
        if self.size_bytes == 0 {
            return false;
        }

        // VRAM delta should roughly match size_bytes
        // Convert size_bytes to MB for comparison
        let expected_delta_mb = (self.size_bytes / (1024 * 1024)) as i64;
        let actual_delta_mb = self.vram_delta_mb as i64;
        let delta_diff = (actual_delta_mb - expected_delta_mb).abs();

        // Allow tolerance for GPU overhead (rounding, fragmentation, etc.)
        delta_diff < Self::DELTA_TOLERANCE_MB
    }

    /// Panic if allocation appears simulated.
    ///
    /// # Panics
    ///
    /// Constitution AP-007 violation with error code EMB-E010 if `is_real()` returns false.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tracking = VramAllocationTracking::new(0x7fff_0000_1000, 104_857_600, 5000, 5100);
    /// tracking.assert_real(); // OK for real data
    ///
    /// // Would panic with "[EMB-E010] SIMULATION_DETECTED: ..."
    /// let fake = VramAllocationTracking {
    ///     base_ptr: 0x7f80_0000_0000, // FAKE POINTER
    ///     ..tracking
    /// };
    /// fake.assert_real(); // PANIC!
    /// ```
    pub fn assert_real(&self) {
        if !self.is_real() {
            panic!(
                "[EMB-E010] SIMULATION_DETECTED: VramAllocationTracking contains fake data. \
                 base_ptr=0x{:x}, size={}, delta={}MB. Constitution AP-007 violation.",
                self.base_ptr, self.size_bytes, self.vram_delta_mb
            );
        }
    }

    /// Get VRAM delta as human-readable string.
    ///
    /// # Returns
    ///
    /// Formatted string showing before/after/delta VRAM usage.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tracking = VramAllocationTracking::new(0x7fff_0000_1000, 104_857_600, 5000, 5100);
    /// assert_eq!(tracking.delta_display(), "100 MB (5000 -> 5100 MB)");
    /// ```
    #[must_use]
    pub fn delta_display(&self) -> String {
        format!(
            "{} MB ({} -> {} MB)",
            self.vram_delta_mb, self.vram_before_mb, self.vram_after_mb
        )
    }

    /// Get allocation size in megabytes.
    #[must_use]
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get allocation size in gigabytes.
    #[must_use]
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}
