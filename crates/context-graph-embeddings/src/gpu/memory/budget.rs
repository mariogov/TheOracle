//! Memory budget configuration for GPU VRAM allocation.

/// Reserved memory pools with fixed budgets.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Model weights budget (e.g., 16GB).
    pub model_weights: usize,
    /// Activation cache budget (e.g., 8GB).
    pub activation_cache: usize,
    /// Working memory budget (e.g., 6GB).
    pub working_memory: usize,
    /// Reserved for system overhead (e.g., 2GB).
    pub reserved: usize,
}

impl MemoryBudget {
    /// Default budget for RTX 5090 32GB.
    pub fn rtx_5090() -> Self {
        const GB: usize = 1024 * 1024 * 1024;
        Self {
            model_weights: 16 * GB,
            activation_cache: 8 * GB,
            working_memory: 6 * GB,
            reserved: 2 * GB,
        }
    }

    /// Total budgeted memory.
    pub fn total(&self) -> usize {
        self.model_weights + self.activation_cache + self.working_memory + self.reserved
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::rtx_5090()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget() {
        let budget = MemoryBudget::rtx_5090();
        assert_eq!(budget.total(), 32 * 1024 * 1024 * 1024);
    }
}
