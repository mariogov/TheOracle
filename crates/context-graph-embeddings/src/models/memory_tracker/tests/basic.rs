//! Basic tests for MemoryTracker.
//!
//! Tests covering:
//! - Construction
//! - Memory allocation

use crate::error::EmbeddingError;
use crate::models::memory_tracker::MemoryTracker;
use crate::types::ModelId;

#[test]
fn test_new_creates_empty_tracker() {
    let tracker = MemoryTracker::new(1_000_000_000);
    assert_eq!(tracker.current_usage(), 0);
    assert_eq!(tracker.budget(), 1_000_000_000);
    assert_eq!(tracker.remaining(), 1_000_000_000);
    assert_eq!(tracker.allocation_count(), 0);
}

#[test]
fn test_allocate_success() {
    let mut tracker = MemoryTracker::new(2_000_000_000);
    let result = tracker.allocate(ModelId::Semantic, 1_400_000_000);

    assert!(result.is_ok());
    assert_eq!(tracker.current_usage(), 1_400_000_000);
    assert_eq!(tracker.allocation_for(ModelId::Semantic), 1_400_000_000);
    assert!(tracker.is_allocated(ModelId::Semantic));
}

#[test]
fn test_allocate_fails_when_budget_exceeded() {
    let mut tracker = MemoryTracker::new(1_000_000_000);
    let result = tracker.allocate(ModelId::Contextual, 1_600_000_000);

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::MemoryBudgetExceeded {
            requested_bytes,
            available_bytes,
            budget_bytes,
        }) => {
            assert_eq!(requested_bytes, 1_600_000_000);
            assert_eq!(available_bytes, 1_000_000_000);
            assert_eq!(budget_bytes, 1_000_000_000);
        }
        _ => panic!("Expected MemoryBudgetExceeded error"),
    }

    // Tracker should be unchanged
    assert_eq!(tracker.current_usage(), 0);
    assert!(!tracker.is_allocated(ModelId::Contextual));
}
