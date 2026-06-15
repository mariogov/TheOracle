//! Task type for automatic profile selection.
//!
//! From teleoplan.md query routing examples:
//! - "How do I implement X?" -> Code search
//! - "Why did X happen?" -> Causal search
//! - "What is similar to X?" -> Semantic search

use serde::{Deserialize, Serialize};

use super::FusionStrategy;

/// Task type for automatic profile selection.
///
/// From teleoplan.md query routing examples:
/// - "How do I implement X?" -> Code search
/// - "Why did X happen?" -> Causal search
/// - "What is similar to X?" -> Semantic search
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TaskType {
    /// Code implementation tasks.
    /// Primary: E6 (Code), E7 (Procedural)
    CodeSearch,

    /// General semantic similarity search.
    /// Primary: E1 (Semantic), E5 (Analogical)
    SemanticSearch,

    /// Temporal/sequence queries.
    /// Primary: E2 (Episodic), E3 (Temporal)
    TemporalSearch,

    /// Causal reasoning queries.
    /// Primary: E4 (Causal), E7 (Procedural)
    CausalSearch,

    /// Factual/knowledge retrieval.
    /// Primary: E12 (Factual), E13 (Sparse)
    FactualSearch,

    /// Social/interpersonal context.
    /// Primary: E9 (Social), E10 (Emotional)
    SocialSearch,

    /// Abstract/conceptual queries.
    /// Primary: E11 (Abstract), E5 (Analogical)
    AbstractSearch,

    /// General balanced search.
    #[default]
    General,
}

impl TaskType {
    /// All task types.
    pub const ALL: [TaskType; 8] = [
        TaskType::CodeSearch,
        TaskType::SemanticSearch,
        TaskType::TemporalSearch,
        TaskType::CausalSearch,
        TaskType::FactualSearch,
        TaskType::SocialSearch,
        TaskType::AbstractSearch,
        TaskType::General,
    ];

    /// Get primary embedder indices for this task type.
    ///
    /// Returns the indices of embedders that should be weighted higher.
    pub fn primary_embedders(self) -> &'static [usize] {
        match self {
            TaskType::CodeSearch => &[5, 6],      // E6, E7
            TaskType::SemanticSearch => &[0, 4],  // E1, E5
            TaskType::TemporalSearch => &[1, 2],  // E2, E3
            TaskType::CausalSearch => &[3, 6],    // E4, E7
            TaskType::FactualSearch => &[11, 12], // E12, E13
            TaskType::SocialSearch => &[8, 9],    // E9, E10
            TaskType::AbstractSearch => &[10, 4], // E11, E5
            TaskType::General => &[0, 3, 11],     // E1, E4, E12 (balanced)
        }
    }

    /// Get secondary embedder indices for this task type.
    pub fn secondary_embedders(self) -> &'static [usize] {
        match self {
            TaskType::CodeSearch => &[3, 11],      // E4, E12
            TaskType::SemanticSearch => &[10, 7],  // E11, E8
            TaskType::TemporalSearch => &[11, 8],  // E12, E9
            TaskType::CausalSearch => &[11, 8],    // E12, E9
            TaskType::FactualSearch => &[0, 3],    // E1, E4
            TaskType::SocialSearch => &[0, 1],     // E1, E2
            TaskType::AbstractSearch => &[0, 3],   // E1, E4
            TaskType::General => &[4, 5, 6, 7, 8], // Middle embedders
        }
    }

    /// Human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            TaskType::CodeSearch => "Code implementation and programming",
            TaskType::SemanticSearch => "General semantic similarity",
            TaskType::TemporalSearch => "Temporal and sequence patterns",
            TaskType::CausalSearch => "Causal reasoning and explanations",
            TaskType::FactualSearch => "Factual knowledge retrieval",
            TaskType::SocialSearch => "Social and interpersonal context",
            TaskType::AbstractSearch => "Abstract and conceptual queries",
            TaskType::General => "Balanced general-purpose search",
        }
    }

    /// Suggested fusion strategy for this task type.
    pub fn suggested_strategy(self) -> FusionStrategy {
        match self {
            TaskType::CodeSearch => FusionStrategy::PrimaryOnly,
            TaskType::SemanticSearch => FusionStrategy::CrossCorrelation,
            TaskType::TemporalSearch => FusionStrategy::Hierarchical,
            TaskType::CausalSearch => FusionStrategy::Hierarchical,
            TaskType::FactualSearch => FusionStrategy::WeightedAverage,
            TaskType::SocialSearch => FusionStrategy::attention_default(),
            TaskType::AbstractSearch => FusionStrategy::CrossCorrelation,
            TaskType::General => FusionStrategy::WeightedAverage,
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::CodeSearch => write!(f, "Code"),
            TaskType::SemanticSearch => write!(f, "Semantic"),
            TaskType::TemporalSearch => write!(f, "Temporal"),
            TaskType::CausalSearch => write!(f, "Causal"),
            TaskType::FactualSearch => write!(f, "Factual"),
            TaskType::SocialSearch => write!(f, "Social"),
            TaskType::AbstractSearch => write!(f, "Abstract"),
            TaskType::General => write!(f, "General"),
        }
    }
}
