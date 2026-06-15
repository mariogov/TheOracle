//! Fusion strategy for combining embeddings in teleological profiles.

use serde::{Deserialize, Serialize};

/// Fusion strategy for combining embeddings in a teleological profile.
///
/// Different strategies optimize for different use cases.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum FusionStrategy {
    /// Simple weighted average of all embeddings.
    #[default]
    WeightedAverage,

    /// Cross-correlation matrix fusion (captures inter-embedding relationships).
    CrossCorrelation,

    /// Tucker tensor decomposition for compact representation.
    /// Ranks: (mode1_rank, mode2_rank, mode3_rank)
    TuckerDecomposition {
        /// Tensor decomposition ranks
        ranks: (usize, usize, usize),
    },

    /// Attention-weighted combination using query context.
    Attention {
        /// Number of attention heads
        heads: usize,
    },

    /// Hierarchical group-then-domain fusion.
    Hierarchical,

    /// Use only primary embeddings (fast path).
    PrimaryOnly,
}

impl FusionStrategy {
    /// Default Tucker ranks from teleoplan.md.
    pub const DEFAULT_TUCKER_RANKS: (usize, usize, usize) = (4, 4, 128);

    /// Default number of attention heads.
    pub const DEFAULT_ATTENTION_HEADS: usize = 4;

    /// Create Tucker decomposition with default ranks.
    pub fn tucker_default() -> Self {
        Self::TuckerDecomposition {
            ranks: Self::DEFAULT_TUCKER_RANKS,
        }
    }

    /// Create attention fusion with default heads.
    pub fn attention_default() -> Self {
        Self::Attention {
            heads: Self::DEFAULT_ATTENTION_HEADS,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            FusionStrategy::WeightedAverage => "Simple weighted average fusion",
            FusionStrategy::CrossCorrelation => "Cross-correlation matrix fusion",
            FusionStrategy::TuckerDecomposition { .. } => "Tucker tensor decomposition",
            FusionStrategy::Attention { .. } => "Attention-weighted fusion",
            FusionStrategy::Hierarchical => "Hierarchical group-then-domain fusion",
            FusionStrategy::PrimaryOnly => "Primary embeddings only (fast)",
        }
    }

    /// Estimated computational cost (1=low, 5=high).
    pub fn cost(&self) -> u8 {
        match self {
            FusionStrategy::PrimaryOnly => 1,
            FusionStrategy::WeightedAverage => 2,
            FusionStrategy::Hierarchical => 2,
            FusionStrategy::CrossCorrelation => 3,
            FusionStrategy::Attention { .. } => 4,
            FusionStrategy::TuckerDecomposition { .. } => 5,
        }
    }
}
