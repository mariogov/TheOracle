//! BatchProcessor statistics.
//!
//! Contains statistics types for tracking batch processing metrics.

use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// INTERNAL STATISTICS
// ============================================================================

/// Internal statistics with atomic counters for thread-safe updates.
#[derive(Debug, Default)]
pub(crate) struct BatchProcessorStatsInternal {
    pub(crate) requests_submitted: AtomicU64,
    pub(crate) batches_processed: AtomicU64,
    pub(crate) requests_completed: AtomicU64,
    pub(crate) requests_failed: AtomicU64,
}

impl BatchProcessorStatsInternal {
    /// Create a snapshot of current statistics.
    pub fn snapshot(&self) -> BatchProcessorStats {
        BatchProcessorStats {
            requests_submitted: self.requests_submitted.load(Ordering::Relaxed),
            batches_processed: self.batches_processed.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.requests_failed.load(Ordering::Relaxed),
            current_queue_depth: 0, // Must be filled by caller
            active_batches: 0,      // Must be filled by caller
        }
    }

    /// Increment requests submitted counter.
    #[inline]
    pub fn inc_requests_submitted(&self) {
        self.requests_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment batches processed counter.
    #[inline]
    pub fn inc_batches_processed(&self) {
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Add to requests completed counter.
    #[inline]
    pub fn add_requests_completed(&self, count: u64) {
        self.requests_completed.fetch_add(count, Ordering::Relaxed);
    }

    /// Add to requests failed counter.
    #[inline]
    pub fn add_requests_failed(&self, count: u64) {
        self.requests_failed.fetch_add(count, Ordering::Relaxed);
    }
}

// ============================================================================
// PUBLIC STATISTICS
// ============================================================================

/// Statistics snapshot for the BatchProcessor.
#[derive(Debug, Clone, Default)]
pub struct BatchProcessorStats {
    /// Total requests submitted.
    pub requests_submitted: u64,
    /// Total batches processed.
    pub batches_processed: u64,
    /// Total requests completed successfully.
    pub requests_completed: u64,
    /// Total requests failed.
    pub requests_failed: u64,
    /// Current queue depth across all models.
    pub current_queue_depth: usize,
    /// Currently processing batch count.
    pub active_batches: usize,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_default() {
        let stats = BatchProcessorStats::default();

        assert_eq!(stats.requests_submitted, 0);
        assert_eq!(stats.batches_processed, 0);
        assert_eq!(stats.requests_completed, 0);
        assert_eq!(stats.requests_failed, 0);
        assert_eq!(stats.current_queue_depth, 0);
        assert_eq!(stats.active_batches, 0);
    }

    #[test]
    fn test_stats_internal_atomic_updates() {
        let stats = BatchProcessorStatsInternal::default();

        stats.inc_requests_submitted();
        stats.inc_requests_submitted();
        stats.inc_requests_submitted();
        stats.inc_requests_submitted();
        stats.inc_requests_submitted();
        stats.inc_batches_processed();
        stats.inc_batches_processed();
        stats.add_requests_completed(4);
        stats.add_requests_failed(1);

        assert_eq!(stats.requests_submitted.load(Ordering::Relaxed), 5);
        assert_eq!(stats.batches_processed.load(Ordering::Relaxed), 2);
        assert_eq!(stats.requests_completed.load(Ordering::Relaxed), 4);
        assert_eq!(stats.requests_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stats_clone() {
        let stats = BatchProcessorStats {
            requests_submitted: 100,
            batches_processed: 10,
            requests_completed: 95,
            requests_failed: 5,
            current_queue_depth: 3,
            active_batches: 2,
        };

        let cloned = stats.clone();

        assert_eq!(stats.requests_submitted, cloned.requests_submitted);
        assert_eq!(stats.batches_processed, cloned.batches_processed);
        assert_eq!(stats.requests_completed, cloned.requests_completed);
        assert_eq!(stats.requests_failed, cloned.requests_failed);
        assert_eq!(stats.current_queue_depth, cloned.current_queue_depth);
        assert_eq!(stats.active_batches, cloned.active_batches);
    }

    #[test]
    fn test_stats_snapshot() {
        let internal = BatchProcessorStatsInternal::default();
        internal.inc_requests_submitted();
        internal.inc_requests_submitted();
        internal.inc_batches_processed();
        internal.add_requests_completed(1);
        internal.add_requests_failed(1);

        let snapshot = internal.snapshot();

        assert_eq!(snapshot.requests_submitted, 2);
        assert_eq!(snapshot.batches_processed, 1);
        assert_eq!(snapshot.requests_completed, 1);
        assert_eq!(snapshot.requests_failed, 1);
        assert_eq!(snapshot.current_queue_depth, 0); // Default, must be set by caller
        assert_eq!(snapshot.active_batches, 0); // Default, must be set by caller
    }
}
