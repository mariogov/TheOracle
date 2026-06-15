//! HNSW index configuration structs and methods.

use serde::{Deserialize, Serialize};

use super::constants::*;
use super::distance::DistanceMetric;

/// HNSW index configuration.
///
/// Parameters from constitution.yaml:
/// - Default: M=16, ef_construction=200, ef_search=100
/// - Matryoshka 128D: M=32, ef_construction=256, ef_search=128
///
/// # Validation
///
/// All parameters are validated on construction:
/// - `m >= 2` (minimum bi-directional links)
/// - `ef_construction >= m` (construction quality)
/// - `ef_search >= 1` (minimum search candidates)
/// - `dimension >= 1` (non-empty vectors)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bi-directional links per node (M parameter).
    /// Higher M = more connections = better recall, more memory.
    pub m: usize,
    /// Size of dynamic candidate list during construction.
    /// Higher ef_construction = better index quality, slower build.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search.
    /// Higher ef_search = better recall, slower queries.
    pub ef_search: usize,
    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,
    /// Embedding dimension for this index.
    pub dimension: usize,
}

impl HnswConfig {
    /// Create config with explicit parameters.
    ///
    /// # Panics
    ///
    /// Panics with "HNSW CONFIG ERROR" if:
    /// - `m < 2`
    /// - `ef_construction < m`
    /// - `ef_search < 1`
    /// - `dimension < 1`
    pub fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: DistanceMetric,
        dimension: usize,
    ) -> Self {
        if m < 2 {
            panic!("HNSW CONFIG ERROR: M must be >= 2, got {}", m);
        }
        if ef_construction < m {
            panic!(
                "HNSW CONFIG ERROR: ef_construction ({}) must be >= M ({})",
                ef_construction, m
            );
        }
        if ef_search < 1 {
            panic!(
                "HNSW CONFIG ERROR: ef_search must be >= 1, got {}",
                ef_search
            );
        }
        if dimension < 1 {
            panic!(
                "HNSW CONFIG ERROR: dimension must be >= 1, got {}",
                dimension
            );
        }

        Self {
            m,
            ef_construction,
            ef_search,
            metric,
            dimension,
        }
    }

    /// Default per-embedder config: M=16, ef_construction=200, ef_search=100.
    ///
    /// From constitution.yaml line 531.
    pub fn default_for_dimension(dimension: usize, metric: DistanceMetric) -> Self {
        Self::new(16, 200, 100, metric, dimension)
    }

    /// E1 Matryoshka 128D config: M=32, ef_construction=256, ef_search=128.
    ///
    /// From constitution.yaml line 524. Optimized for Stage 2 fast filtering.
    pub fn matryoshka_128d() -> Self {
        Self::new(32, 256, 128, DistanceMetric::Cosine, E1_MATRYOSHKA_DIM)
    }

    /// Estimated memory usage per vector in bytes.
    ///
    /// Approximation: dimension * 4 (f32) + M * 2 * 4 (links)
    pub fn estimated_memory_per_vector(&self) -> usize {
        self.dimension * 4 + self.m * 2 * 4
    }
}

/// Configuration for inverted indexes (E6 sparse, E13 SPLADE).
///
/// Used for Stage 1 (SPLADE pre-filter) and legacy E6 sparse slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndexConfig {
    /// Vocabulary size for term IDs (BERT vocab = 30,522).
    pub vocab_size: usize,
    /// Maximum non-zero entries per vector.
    pub max_nnz: usize,
    /// Whether to use BM25 weighting (E13 Stage 1 only).
    pub use_bm25: bool,
}

impl InvertedIndexConfig {
    /// E6 sparse config: 30522 vocab, 1500 max_nnz, no BM25.
    ///
    /// Legacy sparse lexical slot, not used in 5-stage pipeline.
    pub fn e6_sparse() -> Self {
        Self {
            vocab_size: E6_SPARSE_VOCAB,
            max_nnz: 1_500,
            use_bm25: false,
        }
    }

    /// E13 SPLADE config: 30522 vocab, 1500 max_nnz, with BM25.
    ///
    /// Stage 1 recall: BM25 + SPLADE hybrid for initial candidate selection.
    pub fn e13_splade() -> Self {
        Self {
            vocab_size: E13_SPLADE_VOCAB,
            max_nnz: 1_500,
            use_bm25: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_config_new() {
        let cfg = HnswConfig::new(16, 200, 100, DistanceMetric::Cosine, 1024);
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construction, 200);
        assert_eq!(cfg.ef_search, 100);
        assert_eq!(cfg.dimension, 1024);
    }

    #[test]
    #[should_panic(expected = "HNSW CONFIG ERROR")]
    fn test_panic_invalid_m() {
        let _ = HnswConfig::new(1, 200, 100, DistanceMetric::Cosine, 128);
    }

    #[test]
    fn test_minimum_valid_m() {
        let cfg = HnswConfig::new(2, 200, 100, DistanceMetric::Cosine, 128);
        assert_eq!(cfg.m, 2);
    }

    #[test]
    fn test_matryoshka_128d() {
        let cfg = HnswConfig::matryoshka_128d();
        assert_eq!(cfg.dimension, 128);
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.ef_construction, 256);
        assert_eq!(cfg.ef_search, 128);
    }

    #[test]
    fn test_inverted_index_e6() {
        let cfg = InvertedIndexConfig::e6_sparse();
        assert_eq!(cfg.vocab_size, 30_522);
        assert_eq!(cfg.max_nnz, 1_500);
        assert!(!cfg.use_bm25);
    }

    #[test]
    fn test_inverted_index_e13() {
        let cfg = InvertedIndexConfig::e13_splade();
        assert_eq!(cfg.vocab_size, 30_522);
        assert_eq!(cfg.max_nnz, 1_500);
        assert!(cfg.use_bm25);
    }
}
