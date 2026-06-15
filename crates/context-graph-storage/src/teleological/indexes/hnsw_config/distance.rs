//! Distance metric types for vector similarity computation.

use serde::{Deserialize, Serialize};

/// Distance metric for vector similarity computation.
///
/// # Variants
///
/// - `Cosine`: 1 - cos(a, b), range [0, 2]. Most common for normalized embeddings.
/// - `DotProduct`: Inner product. For normalized vectors, equivalent to cosine similarity.
/// - `Euclidean`: L2 distance, range [0, inf). Measures geometric distance.
/// - `AsymmetricCosine`: For E5 causal embeddings where cause->effect != effect->cause.
/// - `MaxSim`: ColBERT-style late interaction. NOT HNSW-compatible (token-level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cos(a, b). Range [0, 2].
    Cosine,
    /// Dot product (inner product). For normalized vectors = cosine similarity.
    DotProduct,
    /// L2 Euclidean distance. Range [0, inf).
    Euclidean,
    /// Asymmetric cosine for E5 causal (cause->effect != effect->cause).
    AsymmetricCosine,
    /// MaxSim for ColBERT late interaction (E12). NOT HNSW-compatible.
    MaxSim,
}

impl DistanceMetric {
    /// Check if this metric is compatible with HNSW indexing.
    ///
    /// MaxSim requires token-level computation and cannot be used with HNSW.
    #[inline]
    pub fn is_hnsw_compatible(&self) -> bool {
        !matches!(self, Self::MaxSim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_hnsw_compatibility() {
        assert!(DistanceMetric::Cosine.is_hnsw_compatible());
        assert!(DistanceMetric::DotProduct.is_hnsw_compatible());
        assert!(DistanceMetric::Euclidean.is_hnsw_compatible());
        assert!(DistanceMetric::AsymmetricCosine.is_hnsw_compatible());
        assert!(
            !DistanceMetric::MaxSim.is_hnsw_compatible(),
            "MaxSim is NOT HNSW-compatible"
        );
    }
}
