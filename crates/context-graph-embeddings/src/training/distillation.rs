//! Online distillation loop: LLM → embedder teaching pipeline.
//!
//! Continuously improves the causal embedder using LLM-validated pairs:
//!
//! ```text
//! LLM discovers relationships → High-confidence pairs → training buffer
//!   → Fine-tune projections (incremental) → Better embeddings
//!   → Better causal search → LLM finds more relationships → repeat
//! ```
//!
//! Key features:
//! - Accumulates LLM-validated pairs (confidence > threshold)
//! - Incremental training when buffer reaches min_pairs
//! - EMA of weights (Polyak averaging) to prevent catastrophic drift
//! - Quality gate: only accept new weights if metrics improve

use std::path::PathBuf;

use super::data::CausalTrainingPair;
use super::evaluation::EvaluationMetrics;

/// Configuration for online distillation.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Minimum number of LLM-validated pairs before incremental training.
    pub min_pairs: usize,
    /// Maximum buffer size before forced training.
    pub max_buffer_size: usize,
    /// Minimum LLM confidence to accept a pair (default: 0.8).
    pub confidence_threshold: f32,
    /// Number of incremental training epochs (default: 10).
    pub incremental_epochs: u32,
    /// EMA coefficient for Polyak averaging (default: 0.999).
    pub ema_tau: f64,
    /// Whether to enforce quality gate (only accept if metrics improve).
    pub quality_gate: bool,
    /// Checkpoint directory for incremental saves.
    pub checkpoint_dir: PathBuf,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            min_pairs: 100,
            max_buffer_size: 10000,
            confidence_threshold: 0.8,
            incremental_epochs: 10,
            ema_tau: 0.999,
            quality_gate: true,
            checkpoint_dir: PathBuf::from("models/causal/trained"),
        }
    }
}

/// Result of an incremental distillation round.
#[derive(Debug, Clone)]
pub struct DistillationResult {
    /// Number of pairs used for training.
    pub num_pairs: usize,
    /// Number of incremental epochs run.
    pub epochs_run: u32,
    /// Metrics after incremental training.
    pub post_metrics: Option<EvaluationMetrics>,
    /// Whether the quality gate passed (new weights accepted).
    pub quality_gate_passed: bool,
    /// Whether weights were rolled back.
    pub rolled_back: bool,
}

/// Online distillation buffer for LLM→embedder teaching.
///
/// Accumulates high-confidence LLM-validated pairs and triggers
/// incremental training when the buffer is large enough.
pub struct DistillationBuffer {
    /// Buffered training pairs.
    buffer: Vec<CausalTrainingPair>,
    /// Configuration.
    config: DistillationConfig,
    /// Total pairs processed across all rounds.
    total_processed: usize,
    /// Number of distillation rounds completed.
    rounds_completed: usize,
    /// Best metrics achieved (for quality gate).
    best_metrics: Option<EvaluationMetrics>,
}

impl DistillationBuffer {
    /// Create a new distillation buffer.
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            buffer: Vec::new(),
            config,
            total_processed: 0,
            rounds_completed: 0,
            best_metrics: None,
        }
    }

    /// Add an LLM-validated pair if it meets the confidence threshold.
    ///
    /// Returns true if the pair was accepted.
    pub fn add_pair(&mut self, pair: CausalTrainingPair) -> bool {
        if pair.confidence < self.config.confidence_threshold {
            return false;
        }

        // Enforce max buffer size with FIFO eviction
        if self.buffer.len() >= self.config.max_buffer_size {
            self.buffer.remove(0);
        }

        self.buffer.push(pair);
        true
    }

    /// Check if the buffer has enough pairs for incremental training.
    pub fn ready_for_training(&self) -> bool {
        self.buffer.len() >= self.config.min_pairs
    }

    /// Drain the buffer for training (returns pairs and clears buffer).
    pub fn drain_for_training(&mut self) -> Vec<CausalTrainingPair> {
        self.total_processed += self.buffer.len();
        self.rounds_completed += 1;
        std::mem::take(&mut self.buffer)
    }

    /// Update best metrics after a successful training round.
    pub fn update_best_metrics(&mut self, metrics: EvaluationMetrics) {
        self.best_metrics = Some(metrics);
    }

    /// Check if new metrics pass the quality gate.
    pub fn passes_quality_gate(&self, new_metrics: &EvaluationMetrics) -> bool {
        if !self.config.quality_gate {
            return true;
        }

        match &self.best_metrics {
            Some(best) => new_metrics.directional_accuracy >= best.directional_accuracy,
            None => true, // No baseline yet, always accept
        }
    }

    /// Get the current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get total pairs processed.
    pub fn total_processed(&self) -> usize {
        self.total_processed
    }

    /// Get number of completed rounds.
    pub fn rounds_completed(&self) -> usize {
        self.rounds_completed
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }

    /// Get the best metrics.
    pub fn best_metrics(&self) -> Option<&EvaluationMetrics> {
        self.best_metrics.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::data::TrainingDirection;

    fn make_pair(confidence: f32) -> CausalTrainingPair {
        CausalTrainingPair::new(
            "cause text".into(),
            "effect text".into(),
            TrainingDirection::Forward,
            confidence,
        )
    }

    #[test]
    fn test_confidence_threshold() {
        let config = DistillationConfig {
            confidence_threshold: 0.8,
            ..Default::default()
        };
        let mut buffer = DistillationBuffer::new(config);

        assert!(!buffer.add_pair(make_pair(0.5)));
        assert!(!buffer.add_pair(make_pair(0.79)));
        assert!(buffer.add_pair(make_pair(0.80)));
        assert!(buffer.add_pair(make_pair(0.95)));

        assert_eq!(buffer.buffer_size(), 2);
    }

    #[test]
    fn test_ready_for_training() {
        let config = DistillationConfig {
            min_pairs: 3,
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let mut buffer = DistillationBuffer::new(config);

        assert!(!buffer.ready_for_training());
        buffer.add_pair(make_pair(0.9));
        buffer.add_pair(make_pair(0.9));
        assert!(!buffer.ready_for_training());
        buffer.add_pair(make_pair(0.9));
        assert!(buffer.ready_for_training());
    }

    #[test]
    fn test_drain_for_training() {
        let config = DistillationConfig {
            min_pairs: 2,
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let mut buffer = DistillationBuffer::new(config);
        buffer.add_pair(make_pair(0.9));
        buffer.add_pair(make_pair(0.8));

        let pairs = buffer.drain_for_training();
        assert_eq!(pairs.len(), 2);
        assert_eq!(buffer.buffer_size(), 0);
        assert_eq!(buffer.total_processed(), 2);
        assert_eq!(buffer.rounds_completed(), 1);
    }

    #[test]
    fn test_quality_gate() {
        let config = DistillationConfig {
            quality_gate: true,
            ..Default::default()
        };
        let mut buffer = DistillationBuffer::new(config);

        // No baseline — always passes
        let metrics_low = EvaluationMetrics {
            directional_accuracy: 0.7,
            ..Default::default()
        };
        assert!(buffer.passes_quality_gate(&metrics_low));

        buffer.update_best_metrics(metrics_low.clone());

        // Higher accuracy passes
        let metrics_high = EvaluationMetrics {
            directional_accuracy: 0.8,
            ..Default::default()
        };
        assert!(buffer.passes_quality_gate(&metrics_high));

        buffer.update_best_metrics(metrics_high.clone());

        // Lower accuracy fails
        let metrics_regression = EvaluationMetrics {
            directional_accuracy: 0.75,
            ..Default::default()
        };
        assert!(!buffer.passes_quality_gate(&metrics_regression));
    }

    #[test]
    fn test_max_buffer_fifo() {
        let config = DistillationConfig {
            max_buffer_size: 3,
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let mut buffer = DistillationBuffer::new(config);

        for i in 0..5 {
            let mut pair = make_pair(0.9);
            pair.cause_text = format!("cause_{}", i);
            buffer.add_pair(pair);
        }

        assert_eq!(buffer.buffer_size(), 3);
        // Oldest entries (0, 1) should have been evicted
        let pairs = buffer.drain_for_training();
        assert_eq!(pairs[0].cause_text, "cause_2");
        assert_eq!(pairs[1].cause_text, "cause_3");
        assert_eq!(pairs[2].cause_text, "cause_4");
    }
}
