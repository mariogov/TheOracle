//! Type definitions for multi-embedder search.
//!
//! Contains all enums, configuration structs, and result types used
//! by the multi-embedder parallel search system.

use std::collections::HashMap;

use uuid::Uuid;

use super::super::super::indexes::EmbedderIndex;
use super::super::result::EmbedderSearchHit;

// ============================================================================
// NORMALIZATION STRATEGIES
// ============================================================================

/// Strategy for normalizing similarity scores across embedders.
///
/// Different embedders produce scores in different ranges and distributions.
/// Normalization makes them comparable before aggregation.
///
/// # Strategies
///
/// - `None`: Use raw similarity scores (0.0-1.0 from cosine)
/// - `MinMax`: Scale to [0, 1] based on result set min/max
/// - `ZScore`: Standardize to zero mean, unit variance
/// - `RankNorm`: Normalize by rank position (1/rank)
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::NormalizationStrategy;
///
/// let strategy = NormalizationStrategy::MinMax;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalizationStrategy {
    /// Use raw similarity scores without normalization.
    /// Suitable when all embedders use same metric and produce similar distributions.
    #[default]
    None,

    /// Min-max normalization: (score - min) / (max - min)
    /// Scales all scores to [0, 1] range within each embedder's result set.
    MinMax,

    /// Z-score normalization: (score - mean) / stddev
    /// Centers scores around 0 with unit variance.
    ZScore,

    /// Rank-based normalization: 1 / rank
    /// First result gets 1.0, second 0.5, third 0.33, etc.
    RankNorm,
}

// ============================================================================
// AGGREGATION STRATEGIES
// ============================================================================

/// Strategy for aggregating scores when an ID appears in multiple embedder results.
///
/// When the same memory ID is found by multiple embedders (e.g., E1Semantic and E8Graph),
/// this determines how to combine their scores into a single final score.
///
/// # Strategies
///
/// - `Max`: Take the highest score from any embedder
/// - `Sum`: Sum all scores (weighted by occurrence count)
/// - `Mean`: Average all scores
/// - `WeightedSum`: Apply embedder-specific weights
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::AggregationStrategy;
///
/// let strategy = AggregationStrategy::Max;
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AggregationStrategy {
    /// Take the maximum score from any embedder.
    /// Good when any strong signal is sufficient evidence.
    #[default]
    Max,

    /// Sum all scores from all embedders.
    /// Rewards IDs that appear in many embedder results.
    Sum,

    /// Average scores across embedders that found this ID.
    /// Balances between signal strength and occurrence count.
    Mean,

    /// Weighted sum with per-embedder weights.
    /// Allows prioritizing certain embedders (e.g., E1Semantic > E8Graph).
    ///
    /// Weights should sum to 1.0 for interpretable scores.
    /// Missing embedders use weight 1.0.
    WeightedSum(HashMap<EmbedderIndex, f32>),
}

// ============================================================================
// MULTI-EMBEDDER SEARCH CONFIGURATION
// ============================================================================

/// Configuration for multi-embedder parallel search.
///
/// # Fields
///
/// - `default_k`: Default number of results per embedder
/// - `default_threshold`: Minimum similarity threshold
/// - `normalization`: Score normalization strategy
/// - `aggregation`: Multi-embedder score aggregation strategy
/// - `max_threads`: Maximum parallel threads (None = rayon default)
/// - `per_embedder_k`: Optional per-embedder k overrides
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::{
///     MultiEmbedderSearchConfig, NormalizationStrategy, AggregationStrategy,
/// };
///
/// let config = MultiEmbedderSearchConfig {
///     default_k: 100,
///     default_threshold: Some(0.5),
///     normalization: NormalizationStrategy::MinMax,
///     aggregation: AggregationStrategy::Max,
///     max_threads: Some(4),
///     per_embedder_k: None,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MultiEmbedderSearchConfig {
    /// Default number of results per embedder.
    pub default_k: usize,

    /// Default minimum similarity threshold.
    pub default_threshold: Option<f32>,

    /// Score normalization strategy.
    pub normalization: NormalizationStrategy,

    /// Score aggregation strategy.
    pub aggregation: AggregationStrategy,

    /// Maximum parallel threads (None = rayon default).
    pub max_threads: Option<usize>,

    /// Per-embedder k overrides.
    pub per_embedder_k: Option<HashMap<EmbedderIndex, usize>>,
}

impl Default for MultiEmbedderSearchConfig {
    fn default() -> Self {
        Self {
            default_k: 100,
            default_threshold: None,
            normalization: NormalizationStrategy::None,
            aggregation: AggregationStrategy::Max,
            max_threads: None,
            per_embedder_k: None,
        }
    }
}

// ============================================================================
// AGGREGATED HIT RESULT
// ============================================================================

/// A single aggregated result from multi-embedder search.
///
/// Contains the final aggregated score and metadata about which embedders
/// contributed to this result.
///
/// # Fields
///
/// - `id`: Memory UUID
/// - `aggregated_score`: Final score after normalization and aggregation
/// - `contributing_embedders`: List of (embedder, original_similarity, normalized_score)
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::AggregatedHit;
/// use context_graph_storage::teleological::indexes::EmbedderIndex;
/// use uuid::Uuid;
///
/// // An ID found by both E1 and E8
/// let hit = AggregatedHit {
///     id: Uuid::new_v4(),
///     aggregated_score: 0.95,
///     contributing_embedders: vec![
///         (EmbedderIndex::E1Semantic, 0.92, 0.95),
///         (EmbedderIndex::E8Graph, 0.88, 0.90),
///     ],
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AggregatedHit {
    /// The memory ID (fingerprint UUID).
    pub id: Uuid,

    /// Final aggregated score after normalization and aggregation.
    pub aggregated_score: f32,

    /// Contributing embedders: (embedder, original_similarity, normalized_score).
    pub contributing_embedders: Vec<(EmbedderIndex, f32, f32)>,
}

impl AggregatedHit {
    /// Get the number of embedders that found this ID.
    #[inline]
    pub fn embedder_count(&self) -> usize {
        self.contributing_embedders.len()
    }

    /// Check if this ID was found by a specific embedder.
    #[inline]
    pub fn found_by(&self, embedder: EmbedderIndex) -> bool {
        self.contributing_embedders
            .iter()
            .any(|(e, _, _)| *e == embedder)
    }

    /// Get the original similarity from a specific embedder (if found).
    #[inline]
    pub fn similarity_from(&self, embedder: EmbedderIndex) -> Option<f32> {
        self.contributing_embedders
            .iter()
            .find(|(e, _, _)| *e == embedder)
            .map(|(_, sim, _)| *sim)
    }

    /// Check if this result has high confidence (score >= 0.9).
    #[inline]
    pub fn is_high_confidence(&self) -> bool {
        self.aggregated_score >= 0.9
    }

    /// Check if this result is multi-modal (found by 2+ embedders).
    #[inline]
    pub fn is_multi_modal(&self) -> bool {
        self.contributing_embedders.len() >= 2
    }
}

// ============================================================================
// PER-EMBEDDER RESULTS
// ============================================================================

/// Results from a single embedder within a multi-embedder search.
///
/// Contains raw hits plus metadata about this embedder's contribution.
#[derive(Debug, Clone)]
pub struct PerEmbedderResults {
    /// Which embedder produced these results.
    pub embedder: EmbedderIndex,

    /// Raw hits from this embedder (pre-normalization).
    pub hits: Vec<EmbedderSearchHit>,

    /// Number of results found.
    pub count: usize,

    /// Search latency for this embedder in microseconds.
    pub latency_us: u64,
}

// ============================================================================
// MULTI-EMBEDDER SEARCH RESULTS
// ============================================================================

/// Results from multi-embedder parallel search.
///
/// Contains both aggregated results and per-embedder breakdown.
///
/// # Fields
///
/// - `aggregated_hits`: Final merged results sorted by aggregated score
/// - `per_embedder`: Raw results from each embedder (before aggregation)
/// - `total_latency_us`: Total wall-clock time including parallelization overhead
/// - `embedders_searched`: Which embedders were queried
/// - `normalization_used`: Normalization strategy applied
/// - `aggregation_used`: Aggregation strategy applied
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::MultiEmbedderSearchResults;
///
/// fn process_results(results: MultiEmbedderSearchResults) {
///     println!("Found {} total results", results.len());
///     println!("Top result: {:?}", results.top());
///
///     // Check per-embedder breakdown
///     for (embedder, per_results) in &results.per_embedder {
///         println!("{:?}: {} hits in {}us",
///                  embedder, per_results.count, per_results.latency_us);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MultiEmbedderSearchResults {
    /// Aggregated hits sorted by aggregated_score descending.
    pub aggregated_hits: Vec<AggregatedHit>,

    /// Per-embedder results (before aggregation).
    pub per_embedder: HashMap<EmbedderIndex, PerEmbedderResults>,

    /// Total wall-clock latency including parallelization.
    pub total_latency_us: u64,

    /// Which embedders were searched.
    pub embedders_searched: Vec<EmbedderIndex>,

    /// Normalization strategy used.
    pub normalization_used: NormalizationStrategy,

    /// Aggregation strategy used.
    pub aggregation_used: AggregationStrategy,
}

impl MultiEmbedderSearchResults {
    /// Check if no aggregated results were found.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.aggregated_hits.is_empty()
    }

    /// Get the number of aggregated results.
    #[inline]
    pub fn len(&self) -> usize {
        self.aggregated_hits.len()
    }

    /// Get the top (highest aggregated score) result.
    #[inline]
    pub fn top(&self) -> Option<&AggregatedHit> {
        self.aggregated_hits.first()
    }

    /// Get all aggregated result IDs.
    #[inline]
    pub fn ids(&self) -> Vec<Uuid> {
        self.aggregated_hits.iter().map(|h| h.id).collect()
    }

    /// Get top N aggregated results.
    #[inline]
    pub fn top_n(&self, n: usize) -> &[AggregatedHit] {
        if n >= self.aggregated_hits.len() {
            &self.aggregated_hits
        } else {
            &self.aggregated_hits[..n]
        }
    }

    /// Get average aggregated score.
    #[inline]
    pub fn average_score(&self) -> Option<f32> {
        if self.aggregated_hits.is_empty() {
            None
        } else {
            let sum: f32 = self
                .aggregated_hits
                .iter()
                .map(|h| h.aggregated_score)
                .sum();
            Some(sum / self.aggregated_hits.len() as f32)
        }
    }

    /// Get iterator over aggregated hits.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &AggregatedHit> {
        self.aggregated_hits.iter()
    }

    /// Get total number of raw hits (before deduplication).
    pub fn total_raw_hits(&self) -> usize {
        self.per_embedder.values().map(|r| r.count).sum()
    }
}
