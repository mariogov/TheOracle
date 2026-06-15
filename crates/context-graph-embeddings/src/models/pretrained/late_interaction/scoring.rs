//! MaxSim scoring methods for ColBERT late-interaction retrieval.

use super::model::LateInteractionModel;
use super::types::TokenEmbeddings;

impl LateInteractionModel {
    /// ColBERT MaxSim scoring: score = Sigma_i max_j cos(q_i, d_j)
    ///
    /// For each query token, find the maximum cosine similarity
    /// to any document token, then sum over all query tokens.
    ///
    /// # Arguments
    /// * `query_tokens` - Query token embeddings
    /// * `doc_tokens` - Document token embeddings
    ///
    /// # Returns
    /// MaxSim score (higher = more similar)
    pub fn maxsim_score(query_tokens: &TokenEmbeddings, doc_tokens: &TokenEmbeddings) -> f32 {
        let mut total_score = 0.0f32;

        for (i, q_vec) in query_tokens.vectors.iter().enumerate() {
            if !query_tokens.mask[i] {
                continue; // Skip padding
            }

            // Find maximum similarity to any document token
            let max_sim = doc_tokens
                .vectors
                .iter()
                .enumerate()
                .filter(|(j, _)| doc_tokens.mask[*j])
                .map(|(_, d_vec)| Self::cosine_similarity(q_vec, d_vec))
                .fold(f32::NEG_INFINITY, f32::max);

            // Only add if we found at least one valid document token
            if max_sim > f32::NEG_INFINITY {
                total_score += max_sim;
            }
        }

        total_score
    }

    /// Batch MaxSim for efficient retrieval.
    ///
    /// # Arguments
    /// * `query_tokens` - Query token embeddings
    /// * `doc_batch` - Batch of document token embeddings
    ///
    /// # Returns
    /// Vector of MaxSim scores, one per document
    pub fn batch_maxsim(query_tokens: &TokenEmbeddings, doc_batch: &[TokenEmbeddings]) -> Vec<f32> {
        doc_batch
            .iter()
            .map(|doc| Self::maxsim_score(query_tokens, doc))
            .collect()
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Assumes both vectors are L2 normalized, so cosine = dot product.
    pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}
