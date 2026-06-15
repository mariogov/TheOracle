//! VRAM allocation tracker for named GPU memory allocations.

use std::collections::HashMap;

use super::error::MemoryError;
use super::stats::MemoryStats;

/// VRAM allocation tracker.
///
/// Tracks named allocations for debugging and monitoring.
#[derive(Debug)]
pub struct VramTracker {
    /// Named allocations.
    allocations: HashMap<String, usize>,
    /// Statistics (includes total_bytes).
    stats: MemoryStats,
}

impl VramTracker {
    /// Create a new VRAM tracker with given capacity.
    ///
    /// # Arguments
    ///
    /// * `total_bytes` - Total VRAM capacity (e.g., 32GB for RTX 5090)
    pub fn new(total_bytes: usize) -> Self {
        Self {
            allocations: HashMap::new(),
            stats: MemoryStats {
                total_bytes,
                ..Default::default()
            },
        }
    }

    /// Create tracker for RTX 5090 (32GB).
    pub fn rtx_5090() -> Self {
        Self::new(32 * 1024 * 1024 * 1024)
    }

    /// Allocate memory with a name for tracking.
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for the allocation (e.g., "semantic_model")
    /// * `bytes` - Number of bytes to allocate
    ///
    /// # Returns
    ///
    /// Ok if allocation succeeded, Err if insufficient memory.
    pub fn allocate(&mut self, name: &str, bytes: usize) -> Result<(), MemoryError> {
        if self.stats.available() < bytes {
            return Err(MemoryError::OutOfMemory {
                requested: bytes,
                available: self.stats.available(),
            });
        }

        self.allocations.insert(name.to_string(), bytes);
        self.stats.allocated_bytes += bytes;
        self.stats.allocation_count += 1;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.allocated_bytes);

        tracing::debug!(
            "GPU allocated '{}': {} bytes ({:.1}% used)",
            name,
            bytes,
            self.stats.utilization_percent()
        );

        Ok(())
    }

    /// Deallocate memory by name.
    ///
    /// # Returns
    ///
    /// Number of bytes freed, or 0 if name not found.
    pub fn deallocate(&mut self, name: &str) -> usize {
        if let Some(bytes) = self.allocations.remove(name) {
            self.stats.allocated_bytes = self.stats.allocated_bytes.saturating_sub(bytes);
            self.stats.deallocation_count += 1;

            tracing::debug!(
                "GPU deallocated '{}': {} bytes ({:.1}% used)",
                name,
                bytes,
                self.stats.utilization_percent()
            );

            bytes
        } else {
            0
        }
    }

    /// Get current memory statistics.
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get available memory in bytes.
    pub fn available(&self) -> usize {
        self.stats.available()
    }

    /// List all allocations.
    pub fn allocations(&self) -> &HashMap<String, usize> {
        &self.allocations
    }

    /// Check if an allocation exists.
    pub fn has_allocation(&self, name: &str) -> bool {
        self.allocations.contains_key(name)
    }

    /// Get size of a specific allocation.
    pub fn allocation_size(&self, name: &str) -> Option<usize> {
        self.allocations.get(name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_tracker() {
        let mut tracker = VramTracker::new(1000);

        // Allocate
        assert!(tracker.allocate("test1", 400).is_ok());
        assert_eq!(tracker.available(), 600);

        // Allocate more
        assert!(tracker.allocate("test2", 300).is_ok());
        assert_eq!(tracker.available(), 300);

        // Fail on over-allocation
        assert!(tracker.allocate("test3", 500).is_err());

        // Deallocate
        assert_eq!(tracker.deallocate("test1"), 400);
        assert_eq!(tracker.available(), 700);
    }
}
