//! Type definitions for teleological matrix search.
//!
//! Contains search strategy enums, comparison scope definitions,
//! and result breakdown structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::groups::GroupType;
use super::super::types::NUM_EMBEDDERS;

/// Search strategy for comparing teleological vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SearchStrategy {
    /// Standard cosine similarity across all components
    #[default]
    Cosine,
    /// Euclidean distance (inverted to similarity)
    Euclidean,
    /// Synergy matrix modulates importance of each pair
    SynergyWeighted,
    /// Compare at group level only (6D)
    GroupHierarchical,
    /// Prioritize cross-correlation patterns (78D)
    CrossCorrelationDominant,
    /// Use Tucker core for compressed comparison (if available)
    TuckerCompressed,
    /// Adaptive: choose best strategy based on vector characteristics
    Adaptive,
}

/// Which components of the teleological vector to compare.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ComparisonScope {
    /// Compare all components: purpose + correlations + groups
    #[default]
    Full,
    /// Compare only the 13D topic profile
    TopicProfileOnly,
    /// Compare only the 78 cross-correlations
    CrossCorrelationsOnly,
    /// Compare only the 6D group alignments
    GroupAlignmentsOnly,
    /// Compare specific embedder pairs
    SpecificPairs(Vec<(usize, usize)>),
    /// Compare specific embedding groups
    SpecificGroups(Vec<GroupType>),
    /// Compare a single embedder's correlation pattern (all 12 pairs it's in)
    SingleEmbedderPattern(usize),
}

/// Detailed breakdown of similarity components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityBreakdown {
    /// Overall similarity score [0, 1]
    pub overall: f32,
    /// Topic profile (13D) similarity [0, 1]
    pub topic_profile: f32,
    /// Cross-correlation similarity [0, 1]
    pub cross_correlations: f32,
    /// Group alignments similarity [0, 1]
    pub group_alignments: f32,
    /// Per-group similarity scores
    pub per_group: HashMap<GroupType, f32>,
    /// Per-embedder topic alignment similarity
    pub per_embedder_topic: [f32; NUM_EMBEDDERS],
    /// Top contributing cross-correlation pairs
    pub top_correlation_pairs: Vec<((usize, usize), f32)>,
    /// Strategy used for this comparison
    pub strategy_used: SearchStrategy,
}

impl Default for SimilarityBreakdown {
    fn default() -> Self {
        Self {
            overall: 0.0,
            topic_profile: 0.0,
            cross_correlations: 0.0,
            group_alignments: 0.0,
            per_group: HashMap::new(),
            per_embedder_topic: [0.0; NUM_EMBEDDERS],
            top_correlation_pairs: Vec::new(),
            strategy_used: SearchStrategy::default(),
        }
    }
}

/// Comprehensive comparison result across all scopes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComprehensiveComparison {
    /// Full comparison with breakdown
    pub full: SimilarityBreakdown,
    /// Topic profile only similarity
    pub topic_profile_only: f32,
    /// Cross-correlations only similarity
    pub correlations_only: f32,
    /// Group alignments only similarity
    pub groups_only: f32,
    /// Per-group similarity
    pub per_group: HashMap<GroupType, f32>,
    /// Per-embedder correlation pattern similarity
    pub per_embedder_pattern: [f32; NUM_EMBEDDERS],
    /// Tucker compressed similarity (if available)
    pub tucker: Option<f32>,
}

impl Default for ComprehensiveComparison {
    fn default() -> Self {
        Self {
            full: SimilarityBreakdown::default(),
            topic_profile_only: 0.0,
            correlations_only: 0.0,
            groups_only: 0.0,
            per_group: HashMap::new(),
            per_embedder_pattern: [0.0; NUM_EMBEDDERS],
            tucker: None,
        }
    }
}
