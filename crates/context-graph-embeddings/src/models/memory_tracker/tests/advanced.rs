//! Advanced tests for MemoryTracker.
//!
//! Tests covering:
//! - Memory deallocation
//! - Usage tracking

use crate::models::memory_tracker::MemoryTracker;
use crate::types::ModelId;

#[test]
fn test_deallocate_success() {
    let mut tracker = MemoryTracker::new(2_000_000_000);
    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();

    let freed = tracker.deallocate(ModelId::Semantic).unwrap();

    assert_eq!(freed, 1_400_000_000);
    assert_eq!(tracker.current_usage(), 0);
    assert!(!tracker.is_allocated(ModelId::Semantic));
    assert_eq!(tracker.allocation_count(), 0);
}

#[test]
fn test_current_usage_accurate() {
    let mut tracker = MemoryTracker::new(10_000_000_000);

    tracker.allocate(ModelId::Semantic, 1_400_000_000).unwrap();
    assert_eq!(tracker.current_usage(), 1_400_000_000);

    tracker.allocate(ModelId::Code, 550_000_000).unwrap();
    assert_eq!(tracker.current_usage(), 1_950_000_000);

    tracker.deallocate(ModelId::Semantic).unwrap();
    assert_eq!(tracker.current_usage(), 550_000_000);
}
