//! Memory statistics for GPU VRAM monitoring.

/// Memory statistics for monitoring.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total VRAM capacity in bytes.
    pub total_bytes: usize,
    /// Currently allocated bytes.
    pub allocated_bytes: usize,
    /// Peak allocation (high water mark).
    pub peak_bytes: usize,
    /// Number of allocations.
    pub allocation_count: usize,
    /// Number of deallocations.
    pub deallocation_count: usize,
}

impl MemoryStats {
    /// Get available memory in bytes.
    pub fn available(&self) -> usize {
        self.total_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Get memory utilization as percentage.
    pub fn utilization_percent(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.allocated_bytes as f32 / self.total_bytes as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            total_bytes: 1000,
            allocated_bytes: 250,
            ..Default::default()
        };

        assert_eq!(stats.available(), 750);
        assert!((stats.utilization_percent() - 25.0).abs() < 0.1);
    }
}
