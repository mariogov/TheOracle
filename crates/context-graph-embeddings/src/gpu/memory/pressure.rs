//! Memory pressure detection for GPU VRAM management.
//!
//! Defines pressure levels based on VRAM utilization to trigger eviction
//! decisions. Critical pressure (>95%) triggers LRU model eviction.
//!
//! # Thresholds
//!
//! | Level | Utilization | Action |
//! |-------|-------------|--------|
//! | Low | <50% | Normal operation |
//! | Medium | 50-80% | Monitor closely |
//! | High | 80-95% | Prepare for eviction |
//! | Critical | >95% | Trigger eviction |

/// Memory pressure levels for eviction decisions.
///
/// Ordered from lowest to highest pressure for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryPressure {
    /// <50% utilization - normal operation, no concerns.
    Low,
    /// 50-80% utilization - monitor closely.
    Medium,
    /// 80-95% utilization - prepare for eviction.
    High,
    /// >95% utilization - MUST evict least-recently-used model.
    Critical,
}

impl MemoryPressure {
    /// Pressure threshold percentages.
    pub const LOW_THRESHOLD: f32 = 50.0;
    pub const MEDIUM_THRESHOLD: f32 = 80.0;
    pub const HIGH_THRESHOLD: f32 = 95.0;

    /// Calculate pressure level from utilization percentage.
    ///
    /// # Arguments
    ///
    /// * `percent` - VRAM utilization as a percentage (0.0 to 100.0)
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_embeddings::gpu::MemoryPressure;
    ///
    /// assert_eq!(MemoryPressure::from_utilization(25.0), MemoryPressure::Low);
    /// assert_eq!(MemoryPressure::from_utilization(65.0), MemoryPressure::Medium);
    /// assert_eq!(MemoryPressure::from_utilization(90.0), MemoryPressure::High);
    /// assert_eq!(MemoryPressure::from_utilization(98.0), MemoryPressure::Critical);
    /// ```
    #[inline]
    pub fn from_utilization(percent: f32) -> Self {
        match percent {
            p if p < Self::LOW_THRESHOLD => MemoryPressure::Low,
            p if p < Self::MEDIUM_THRESHOLD => MemoryPressure::Medium,
            p if p < Self::HIGH_THRESHOLD => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        }
    }

    /// Calculate pressure level from bytes used and total budget.
    ///
    /// # Arguments
    ///
    /// * `used_bytes` - Currently allocated bytes
    /// * `total_bytes` - Total budget bytes
    ///
    /// # Returns
    ///
    /// The appropriate pressure level based on utilization.
    #[inline]
    pub fn from_bytes(used_bytes: usize, total_bytes: usize) -> Self {
        if total_bytes == 0 {
            return MemoryPressure::Critical;
        }
        let percent = (used_bytes as f32 / total_bytes as f32) * 100.0;
        Self::from_utilization(percent)
    }

    /// Whether eviction should be triggered at this pressure level.
    ///
    /// Only `Critical` pressure triggers eviction.
    #[inline]
    pub fn should_evict(&self) -> bool {
        *self == MemoryPressure::Critical
    }

    /// Whether allocation should proceed with caution.
    ///
    /// Returns true for `High` and `Critical` pressure.
    #[inline]
    pub fn should_warn(&self) -> bool {
        matches!(self, MemoryPressure::High | MemoryPressure::Critical)
    }

    /// Get a human-readable description of the pressure level.
    pub fn description(&self) -> &'static str {
        match self {
            MemoryPressure::Low => "Normal operation (<50% utilization)",
            MemoryPressure::Medium => "Monitor closely (50-80% utilization)",
            MemoryPressure::High => "Prepare for eviction (80-95% utilization)",
            MemoryPressure::Critical => "EVICTION REQUIRED (>95% utilization)",
        }
    }
}

impl std::fmt::Display for MemoryPressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryPressure::Low => write!(f, "Low"),
            MemoryPressure::Medium => write!(f, "Medium"),
            MemoryPressure::High => write!(f, "High"),
            MemoryPressure::Critical => write!(f, "Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_from_utilization_low() {
        assert_eq!(MemoryPressure::from_utilization(0.0), MemoryPressure::Low);
        assert_eq!(MemoryPressure::from_utilization(25.0), MemoryPressure::Low);
        assert_eq!(MemoryPressure::from_utilization(49.9), MemoryPressure::Low);
        println!("[PASS] MemoryPressure::Low at <50% utilization");
    }

    #[test]
    fn test_pressure_from_utilization_medium() {
        assert_eq!(
            MemoryPressure::from_utilization(50.0),
            MemoryPressure::Medium
        );
        assert_eq!(
            MemoryPressure::from_utilization(65.0),
            MemoryPressure::Medium
        );
        assert_eq!(
            MemoryPressure::from_utilization(79.9),
            MemoryPressure::Medium
        );
        println!("[PASS] MemoryPressure::Medium at 50-80% utilization");
    }

    #[test]
    fn test_pressure_from_utilization_high() {
        assert_eq!(MemoryPressure::from_utilization(80.0), MemoryPressure::High);
        assert_eq!(MemoryPressure::from_utilization(90.0), MemoryPressure::High);
        assert_eq!(MemoryPressure::from_utilization(94.9), MemoryPressure::High);
        println!("[PASS] MemoryPressure::High at 80-95% utilization");
    }

    #[test]
    fn test_pressure_from_utilization_critical() {
        assert_eq!(
            MemoryPressure::from_utilization(95.0),
            MemoryPressure::Critical
        );
        assert_eq!(
            MemoryPressure::from_utilization(99.0),
            MemoryPressure::Critical
        );
        assert_eq!(
            MemoryPressure::from_utilization(100.0),
            MemoryPressure::Critical
        );
        println!("[PASS] MemoryPressure::Critical at >95% utilization");
    }

    #[test]
    fn test_pressure_from_bytes() {
        // 50 out of 100 = 50%
        assert_eq!(MemoryPressure::from_bytes(50, 100), MemoryPressure::Medium);
        // 95 out of 100 = 95%
        assert_eq!(
            MemoryPressure::from_bytes(95, 100),
            MemoryPressure::Critical
        );
        // Zero total = Critical (edge case)
        assert_eq!(MemoryPressure::from_bytes(0, 0), MemoryPressure::Critical);
        println!("[PASS] MemoryPressure::from_bytes() calculates correctly");
    }

    #[test]
    fn test_should_evict() {
        assert!(!MemoryPressure::Low.should_evict());
        assert!(!MemoryPressure::Medium.should_evict());
        assert!(!MemoryPressure::High.should_evict());
        assert!(MemoryPressure::Critical.should_evict());
        println!("[PASS] should_evict() returns true only for Critical");
    }

    #[test]
    fn test_should_warn() {
        assert!(!MemoryPressure::Low.should_warn());
        assert!(!MemoryPressure::Medium.should_warn());
        assert!(MemoryPressure::High.should_warn());
        assert!(MemoryPressure::Critical.should_warn());
        println!("[PASS] should_warn() returns true for High and Critical");
    }

    #[test]
    fn test_pressure_ordering() {
        assert!(MemoryPressure::Low < MemoryPressure::Medium);
        assert!(MemoryPressure::Medium < MemoryPressure::High);
        assert!(MemoryPressure::High < MemoryPressure::Critical);
        println!("[PASS] MemoryPressure ordering is Low < Medium < High < Critical");
    }

    #[test]
    fn test_pressure_display() {
        assert_eq!(format!("{}", MemoryPressure::Low), "Low");
        assert_eq!(format!("{}", MemoryPressure::Medium), "Medium");
        assert_eq!(format!("{}", MemoryPressure::High), "High");
        assert_eq!(format!("{}", MemoryPressure::Critical), "Critical");
        println!("[PASS] Display trait works correctly");
    }

    #[test]
    fn test_threshold_boundary_exact() {
        // Test exact boundary values
        // Note: Using values well within each range to avoid f32 precision issues
        assert_eq!(MemoryPressure::from_utilization(49.0), MemoryPressure::Low);
        assert_eq!(
            MemoryPressure::from_utilization(50.0),
            MemoryPressure::Medium
        );
        assert_eq!(
            MemoryPressure::from_utilization(79.0),
            MemoryPressure::Medium
        );
        assert_eq!(MemoryPressure::from_utilization(80.0), MemoryPressure::High);
        assert_eq!(MemoryPressure::from_utilization(94.0), MemoryPressure::High);
        assert_eq!(
            MemoryPressure::from_utilization(95.0),
            MemoryPressure::Critical
        );
        println!("[PASS] Exact boundary values are classified correctly");
    }
}
