//! Combined memory pools for warm model loading.
//!
//! This module provides the unified `WarmMemoryPools` interface that manages
//! both the non-evictable model pool and the evictable working memory pool.

use super::model_pool::ModelMemoryPool;
use super::types::ModelAllocation;
use super::working_pool::WorkingMemoryPool;
use crate::warm::config::WarmConfig;
use crate::warm::error::WarmResult;

/// Combined memory pools for warm model loading.
///
/// Manages two isolated pools:
/// - **Model Pool**: Non-evictable, for permanent model weights
/// - **Working Pool**: Evictable, for inference activations
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::warm::memory_pool::WarmMemoryPools;
///
/// let mut pools = WarmMemoryPools::rtx_5090();
///
/// // Load a model (non-evictable)
/// pools.allocate_model("E1_Semantic", 800_000_000, 0x1000)?;
///
/// // Allocate working memory (evictable)
/// pools.allocate_working(50_000_000)?;
///
/// // Check budget compliance
/// assert!(pools.is_within_budget());
/// ```
#[derive(Debug, Clone)]
pub struct WarmMemoryPools {
    /// Non-evictable pool for model weights.
    pub(super) model_pool: ModelMemoryPool,
    /// Evictable pool for working memory.
    pub(super) working_pool: WorkingMemoryPool,
    /// Configuration that created these pools.
    config: WarmConfig,
}

impl WarmMemoryPools {
    /// Create new memory pools from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying pool sizes
    ///
    /// # Pool Sizing
    ///
    /// - Model pool capacity: `config.vram_budget_bytes`
    /// - Working pool capacity: `config.vram_headroom_bytes`
    #[must_use]
    pub fn new(config: WarmConfig) -> Self {
        let model_pool = ModelMemoryPool::new(config.vram_budget_bytes);
        let working_pool = WorkingMemoryPool::new(config.vram_headroom_bytes);

        Self {
            model_pool,
            working_pool,
            config,
        }
    }

    /// Create pools sized for RTX 5090 (32GB VRAM).
    ///
    /// Pool allocation:
    /// - Model pool: 24GB (non-evictable)
    /// - Working pool: 8GB (evictable)
    #[must_use]
    pub fn rtx_5090() -> Self {
        Self::new(WarmConfig::default())
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &WarmConfig {
        &self.config
    }

    /// Allocate memory for a model in the non-evictable pool.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Unique identifier for the model
    /// * `size_bytes` - Size of allocation in bytes
    /// * `vram_ptr` - Raw VRAM pointer from CUDA allocation
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelAlreadyRegistered` if model already allocated.
    /// Returns `WarmError::VramAllocationFailed` if allocation exceeds model pool capacity.
    pub fn allocate_model(
        &mut self,
        model_id: &str,
        size_bytes: usize,
        vram_ptr: u64,
    ) -> WarmResult<()> {
        self.model_pool.allocate(model_id, size_bytes, vram_ptr)
    }

    /// Free a model allocation from the non-evictable pool.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier of model to free
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelNotRegistered` if model not found.
    pub fn free_model(&mut self, model_id: &str) -> WarmResult<()> {
        self.model_pool.free(model_id)?;
        Ok(())
    }

    /// Allocate working memory from the evictable pool.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size of allocation in bytes
    ///
    /// # Errors
    ///
    /// Returns `WarmError::WorkingMemoryExhausted` if allocation exceeds working pool capacity.
    pub fn allocate_working(&mut self, size_bytes: usize) -> WarmResult<()> {
        self.working_pool.allocate(size_bytes)
    }

    /// Free working memory from the evictable pool.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size to free in bytes
    ///
    /// # Note
    ///
    /// This method always succeeds. If the amount to free exceeds what
    /// is allocated, only the allocated amount will be freed.
    pub fn free_working(&mut self, size_bytes: usize) -> WarmResult<()> {
        self.working_pool.free(size_bytes);
        Ok(())
    }

    /// Get available bytes in the model pool.
    #[must_use]
    pub fn available_model_bytes(&self) -> usize {
        self.model_pool.available()
    }

    /// Get available bytes in the working pool.
    #[must_use]
    pub fn available_working_bytes(&self) -> usize {
        self.working_pool.available()
    }

    /// Get total allocated bytes across both pools.
    #[must_use]
    pub fn total_allocated_bytes(&self) -> usize {
        self.model_pool
            .allocated()
            .saturating_add(self.working_pool.allocated())
    }

    /// Check if allocations are within the configured budget.
    ///
    /// Returns `true` if:
    /// - Model pool is within `vram_budget_bytes`
    /// - Working pool is within `vram_headroom_bytes`
    #[must_use]
    pub fn is_within_budget(&self) -> bool {
        self.model_pool.allocated() <= self.config.vram_budget_bytes
            && self.working_pool.allocated() <= self.config.vram_headroom_bytes
    }

    /// Get allocation info for a specific model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier of model to look up
    ///
    /// # Returns
    ///
    /// `Some(&ModelAllocation)` if found, `None` otherwise.
    #[must_use]
    pub fn get_model_allocation(&self, model_id: &str) -> Option<&ModelAllocation> {
        self.model_pool.get(model_id)
    }

    /// Get all current model allocations.
    #[must_use]
    pub fn list_model_allocations(&self) -> &[ModelAllocation] {
        self.model_pool.allocations()
    }

    /// Get the model pool capacity.
    #[must_use]
    pub fn model_pool_capacity(&self) -> usize {
        self.model_pool.capacity()
    }

    /// Get the working pool capacity.
    #[must_use]
    pub fn working_pool_capacity(&self) -> usize {
        self.working_pool.capacity()
    }

    /// Get total capacity across both pools.
    #[must_use]
    pub fn total_capacity(&self) -> usize {
        self.model_pool
            .capacity()
            .saturating_add(self.working_pool.capacity())
    }

    /// Get utilization percentage (0.0 - 1.0) for model pool.
    #[must_use]
    pub fn model_pool_utilization(&self) -> f64 {
        if self.model_pool.capacity() == 0 {
            return 0.0;
        }
        self.model_pool.allocated() as f64 / self.model_pool.capacity() as f64
    }

    /// Get utilization percentage (0.0 - 1.0) for working pool.
    #[must_use]
    pub fn working_pool_utilization(&self) -> f64 {
        if self.working_pool.capacity() == 0 {
            return 0.0;
        }
        self.working_pool.allocated() as f64 / self.working_pool.capacity() as f64
    }

    /// Reset the working memory pool.
    ///
    /// Useful for clearing between inference batches or recovering
    /// from memory pressure.
    pub fn reset_working_pool(&mut self) {
        self.working_pool.reset();
    }
}
