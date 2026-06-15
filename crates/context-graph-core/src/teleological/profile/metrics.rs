//! Performance metrics for teleological profiles.

use serde::{Deserialize, Serialize};

/// Performance metrics for a teleological profile.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ProfileMetrics {
    /// Mean Reciprocal Rank (position of first relevant result).
    pub mrr: f32,

    /// Recall at position 5.
    pub recall_at_5: f32,

    /// Recall at position 10.
    pub recall_at_10: f32,

    /// Precision at position 5.
    pub precision_at_5: f32,

    /// Precision at position 10.
    pub precision_at_10: f32,

    /// Number of retrievals used to compute these metrics.
    pub retrieval_count: u64,

    /// Average latency in milliseconds.
    pub avg_latency_ms: f32,
}

impl ProfileMetrics {
    /// Create metrics with all values.
    pub fn new(
        mrr: f32,
        recall_at_5: f32,
        recall_at_10: f32,
        precision_at_5: f32,
        precision_at_10: f32,
    ) -> Self {
        Self {
            mrr,
            recall_at_5,
            recall_at_10,
            precision_at_5,
            precision_at_10,
            retrieval_count: 0,
            avg_latency_ms: 0.0,
        }
    }

    /// Overall quality score (weighted combination of metrics).
    pub fn quality_score(&self) -> f32 {
        // MRR (30%) + Recall@10 (30%) + Precision@10 (40%)
        0.3 * self.mrr + 0.3 * self.recall_at_10 + 0.4 * self.precision_at_10
    }

    /// F1 score at position 10.
    pub fn f1_at_10(&self) -> f32 {
        let p = self.precision_at_10;
        let r = self.recall_at_10;

        if p + r < f32::EPSILON {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Update metrics with EWMA.
    #[allow(clippy::too_many_arguments)]
    pub fn update_ewma(
        &mut self,
        new_mrr: f32,
        new_recall_5: f32,
        new_recall_10: f32,
        new_precision_5: f32,
        new_precision_10: f32,
        latency_ms: f32,
        alpha: f32,
    ) {
        self.mrr = alpha * new_mrr + (1.0 - alpha) * self.mrr;
        self.recall_at_5 = alpha * new_recall_5 + (1.0 - alpha) * self.recall_at_5;
        self.recall_at_10 = alpha * new_recall_10 + (1.0 - alpha) * self.recall_at_10;
        self.precision_at_5 = alpha * new_precision_5 + (1.0 - alpha) * self.precision_at_5;
        self.precision_at_10 = alpha * new_precision_10 + (1.0 - alpha) * self.precision_at_10;
        self.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.avg_latency_ms;
        self.retrieval_count += 1;
    }
}
