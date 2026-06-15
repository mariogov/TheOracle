//! Configuration for teleological matrix search.
//!
//! Defines the MatrixSearchConfig struct and associated presets.

use serde::{Deserialize, Serialize};

use super::super::synergy_matrix::SynergyMatrix;
use super::types::{ComparisonScope, SearchStrategy};
use super::weights::ComponentWeights;

/// Configuration for teleological matrix search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatrixSearchConfig {
    /// Search strategy to use
    pub strategy: SearchStrategy,
    /// Which components to compare
    pub scope: ComparisonScope,
    /// Component weights (for Full scope)
    pub weights: ComponentWeights,
    /// Optional synergy matrix for weighted comparisons
    pub synergy_matrix: Option<SynergyMatrix>,
    /// Minimum similarity threshold for results
    pub min_similarity: f32,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Whether to compute per-component breakdown
    pub compute_breakdown: bool,
}

impl Default for MatrixSearchConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::default(),
            scope: ComparisonScope::default(),
            weights: ComponentWeights::default(),
            synergy_matrix: None,
            min_similarity: 0.0,
            max_results: 100,
            compute_breakdown: false,
        }
    }
}

impl MatrixSearchConfig {
    /// Create config for correlation-focused search
    pub fn correlation_focused() -> Self {
        Self {
            strategy: SearchStrategy::CrossCorrelationDominant,
            scope: ComparisonScope::CrossCorrelationsOnly,
            weights: ComponentWeights::correlation_focused(),
            ..Default::default()
        }
    }

    /// Create config for topic-profile-focused search
    pub fn topic_profile_focused() -> Self {
        Self {
            strategy: SearchStrategy::Cosine,
            scope: ComparisonScope::TopicProfileOnly,
            weights: ComponentWeights::topic_focused(),
            ..Default::default()
        }
    }

    /// Create config for group-hierarchical search
    pub fn group_hierarchical() -> Self {
        Self {
            strategy: SearchStrategy::GroupHierarchical,
            scope: ComparisonScope::GroupAlignmentsOnly,
            weights: ComponentWeights::group_focused(),
            ..Default::default()
        }
    }

    /// Create config with synergy weighting
    pub fn with_synergy(synergy_matrix: SynergyMatrix) -> Self {
        Self {
            strategy: SearchStrategy::SynergyWeighted,
            synergy_matrix: Some(synergy_matrix),
            ..Default::default()
        }
    }

    /// Create config for adaptive search
    pub fn adaptive() -> Self {
        Self {
            strategy: SearchStrategy::Adaptive,
            compute_breakdown: true,
            ..Default::default()
        }
    }
}
