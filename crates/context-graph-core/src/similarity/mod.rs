//! Cross-Space Similarity Engine for multi-embedding similarity computation.
//!
//! This module provides the `CrossSpaceSimilarityEngine` trait and `DefaultCrossSpaceEngine`
//! implementation for computing unified similarity scores across 13 embedding spaces.
//!
//! # Architecture (from constitution.yaml)
//!
//! The similarity engine supports multiple aggregation strategies:
//! - **RRF (Primary)**: Reciprocal Rank Fusion with k=60
//! - **Weighted Average**: Static or purpose-aligned weights
//! - **MaxPooling**: Maximum similarity across spaces
//! - **Late Interaction**: MaxSim for E12 ColBERT embeddings
//!
//! # Performance Requirements
//!
//! - Pair similarity: **<5ms**
//! - Batch 100: **<50ms**
//! - RRF fusion: **<2ms** per 1000 candidates
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_core::similarity::{
//!     CrossSpaceSimilarityEngine, DefaultCrossSpaceEngine,
//!     CrossSpaceConfig, WeightingStrategy,
//! };
//!
//! let engine = DefaultCrossSpaceEngine::new();
//! let config = CrossSpaceConfig::default(); // Uses RRF with k=60
//!
//! let result = engine.compute_similarity(&fp1, &fp2, &config).await?;
//! println!("Similarity: {:.4}", result.score);
//! ```
//!
//! # Module Structure
//!
//! - `config`: Configuration types (`CrossSpaceConfig`, `WeightingStrategy`)
//! - `result`: Result types (`CrossSpaceSimilarity`)
//! - `error`: Error types (`SimilarityError`)
//! - `engine`: Core trait (`CrossSpaceSimilarityEngine`)
//! - `default_engine`: Default implementation (`DefaultCrossSpaceEngine`)
//! - `multi_utl`: Multi-UTL formula implementation
//! - `explanation`: Human-readable explanations

mod config;
mod default_engine;
mod dense;
mod engine;
mod error;
mod explanation;
mod multi_utl;
mod result;
mod sparse;
mod token_level;

#[cfg(test)]
mod tests;

// Re-export public types
pub use config::{CrossSpaceConfig, MissingSpaceHandling, WeightingStrategy};
pub use default_engine::DefaultCrossSpaceEngine;
#[cfg(target_arch = "x86_64")]
pub use dense::cosine_similarity_simd;
pub use dense::{
    cosine_similarity, dot_product, euclidean_distance, l2_norm, normalize, DenseSimilarityError,
};
pub use engine::CrossSpaceSimilarityEngine;
pub use error::SimilarityError;
pub use explanation::SimilarityExplanation;
pub use multi_utl::{sigmoid, MultiUtlParams};
pub use result::CrossSpaceSimilarity;

// Sparse similarity functions for E6/E13 (SPLADE) embeddings
pub use sparse::{jaccard_similarity, SparseSimilarityError};

// Token-level similarity functions for E12 (ColBERT MaxSim) embeddings
pub use token_level::{
    approximate_max_sim, max_sim, symmetric_max_sim, token_alignments, TokenAlignment,
};
