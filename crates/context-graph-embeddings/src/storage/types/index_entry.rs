//! Index entry type for per-embedder HNSW indexes.

use uuid::Uuid;

/// Entry in a per-embedder HNSW index.
///
/// This type is used for INDEXING in layer2c_per_embedder (13× HNSW).
/// Contains DEQUANTIZED vectors for similarity search.
///
/// # Usage in 5-Stage Pipeline
/// Stage 3 (Multi-space rerank): Query each HNSW index → get IndexEntry results → RRF fusion
///
/// # Memory Consideration
/// IndexEntry holds dequantized f32 vectors, so it's memory-intensive.
/// Don't hold large collections in memory - use for query-time only.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// UUID of the fingerprint this entry belongs to.
    pub id: Uuid,

    /// Embedder index this entry is from (0-12).
    pub embedder_idx: u8,

    /// Dequantized embedding vector (full precision f32).
    /// Length depends on embedder:
    /// - E1: 1024, E2-E4: 512 each, E5: 768, E6: sparse, E7: 1536
    /// - E8: 1024, E9: 1024 (from 10K binary), E10: 768, E11: 768
    /// - E12: 128 per token, E13: sparse
    pub vector: Vec<f32>,

    /// Precomputed L2 norm for fast cosine similarity.
    /// norm = sqrt(sum(x_i^2))
    pub norm: f32,
}

impl IndexEntry {
    /// Create index entry with precomputed norm.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    /// * `embedder_idx` - Which embedder (0-12)
    /// * `vector` - Dequantized embedding vector
    #[must_use]
    pub fn new(id: Uuid, embedder_idx: u8, vector: Vec<f32>) -> Self {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self {
            id,
            embedder_idx,
            vector,
            norm,
        }
    }

    /// Get normalized vector for cosine similarity.
    ///
    /// # Returns
    /// Unit vector (L2 norm = 1.0), or zero vector if norm is ~0.
    #[must_use]
    pub fn normalized(&self) -> Vec<f32> {
        if self.norm > 1e-10 {
            self.vector.iter().map(|x| x / self.norm).collect()
        } else {
            vec![0.0; self.vector.len()]
        }
    }

    /// Compute cosine similarity with another vector.
    ///
    /// # Arguments
    /// * `other` - Query vector (must have same length)
    ///
    /// # Returns
    /// Cosine similarity in range [-1.0, 1.0]
    ///
    /// # Panics
    /// Panics if vector lengths don't match.
    #[must_use]
    pub fn cosine_similarity(&self, other: &[f32]) -> f32 {
        if self.vector.len() != other.len() {
            panic!(
                "SIMILARITY ERROR: Vector length mismatch. Entry has {} dims, query has {} dims. \
                 Embedder index: {}. This indicates dimension mismatch bug.",
                self.vector.len(),
                other.len(),
                self.embedder_idx
            );
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();
        let other_norm: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();

        if self.norm > 1e-10 && other_norm > 1e-10 {
            dot / (self.norm * other_norm)
        } else {
            0.0
        }
    }
}
