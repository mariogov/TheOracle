//! Evictable memory pool for working memory (inference activations).
//!
//! This module provides the `WorkingMemoryPool` which holds temporary
//! allocations for inference operations. Unlike the model pool, these
//! allocations CAN be reclaimed under memory pressure.

use crate::warm::error::{WarmError, WarmResult};

/// Evictable memory pool for working memory (inference activations).
///
/// This pool holds temporary allocations for inference operations.
/// Unlike the model pool, these allocations CAN be reclaimed under
/// memory pressure.
///
/// # Invariants
///
/// - `allocated_bytes <= capacity_bytes` (enforced by allocation methods)
#[derive(Debug, Clone)]
pub struct WorkingMemoryPool {
    /// Total capacity in bytes (e.g., 8GB for RTX 5090).
    capacity_bytes: usize,
    /// Currently allocated bytes for working memory.
    allocated_bytes: usize,
}

impl WorkingMemoryPool {
    /// Create a new working memory pool with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity_bytes` - Maximum bytes for working memory
    #[must_use]
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            allocated_bytes: 0,
        }
    }

    /// Get the total capacity of this pool in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    /// Get the currently allocated bytes.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated_bytes
    }

    /// Get the available bytes for new allocations.
    #[must_use]
    pub fn available(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Allocate working memory.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size of allocation in bytes
    ///
    /// # Errors
    ///
    /// Returns `WarmError::WorkingMemoryExhausted` if allocation exceeds capacity.
    pub fn allocate(&mut self, size_bytes: usize) -> WarmResult<()> {
        // Check capacity - NO FALLBACKS
        if self.allocated_bytes.saturating_add(size_bytes) > self.capacity_bytes {
            return Err(WarmError::WorkingMemoryExhausted {
                requested_bytes: size_bytes,
                available_bytes: self.available(),
            });
        }

        self.allocated_bytes = self.allocated_bytes.saturating_add(size_bytes);
        Ok(())
    }

    /// Free working memory.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size to free in bytes
    ///
    /// # Note
    ///
    /// This method silently caps the freed amount to prevent underflow.
    /// If you attempt to free more than is allocated, only the allocated
    /// amount will be freed.
    pub fn free(&mut self, size_bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size_bytes);
    }

    /// Reset all working memory allocations.
    ///
    /// This is useful for clearing working memory between inference batches
    /// or when recovering from memory pressure.
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
    }
}
