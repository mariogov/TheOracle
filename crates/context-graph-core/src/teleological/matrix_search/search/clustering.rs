//! Clustering and centroid computation for teleological vectors.
//!
//! Contains methods for pairwise similarity matrices, clustering,
//! and computing vector centroids.

use super::super::super::groups::{GroupAlignments, NUM_GROUPS};
use super::super::super::synergy_matrix::CROSS_CORRELATION_COUNT;
use super::super::super::types::NUM_EMBEDDERS;
use super::super::super::vector::TeleologicalVector;
use super::core::TeleologicalMatrixSearch;

impl TeleologicalMatrixSearch {
    /// Compute pairwise similarity matrix for a collection.
    ///
    /// Returns N x N matrix where [i][j] = similarity(vectors[i], vectors[j]).
    #[allow(clippy::needless_range_loop)]
    pub fn pairwise_similarity_matrix(&self, vectors: &[TeleologicalVector]) -> Vec<Vec<f32>> {
        let n = vectors.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0; // Self-similarity
            for j in (i + 1)..n {
                let sim = self.similarity(&vectors[i], &vectors[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim; // Symmetric
            }
        }

        matrix
    }

    /// Find clusters of similar vectors.
    ///
    /// Returns groups of vector indices that are mutually similar (above threshold).
    #[allow(clippy::needless_range_loop)]
    pub fn find_clusters(
        &self,
        vectors: &[TeleologicalVector],
        similarity_threshold: f32,
    ) -> Vec<Vec<usize>> {
        let n = vectors.len();
        let sim_matrix = self.pairwise_similarity_matrix(vectors);

        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }

            // Start new cluster with this vector
            let mut cluster = vec![i];
            visited[i] = true;

            // Find all vectors similar to any in the cluster
            let mut frontier = vec![i];
            while let Some(current) = frontier.pop() {
                for (j, v) in visited.iter_mut().enumerate() {
                    if !*v && sim_matrix[current][j] >= similarity_threshold {
                        *v = true;
                        cluster.push(j);
                        frontier.push(j);
                    }
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    /// Compute centroid of a set of teleological vectors.
    pub fn compute_centroid(&self, vectors: &[TeleologicalVector]) -> TeleologicalVector {
        if vectors.is_empty() {
            return TeleologicalVector::default();
        }

        let n = vectors.len() as f32;

        // Average topic profiles
        let mut avg_alignments = [0.0f32; NUM_EMBEDDERS];
        for v in vectors {
            for (avg, &val) in avg_alignments
                .iter_mut()
                .zip(v.topic_profile.alignments.iter())
            {
                *avg += val;
            }
        }
        for avg in avg_alignments.iter_mut() {
            *avg /= n;
        }

        // Average cross-correlations
        let mut avg_correlations = vec![0.0f32; CROSS_CORRELATION_COUNT];
        for v in vectors {
            for (avg, &val) in avg_correlations.iter_mut().zip(v.cross_correlations.iter()) {
                *avg += val;
            }
        }
        for avg in avg_correlations.iter_mut() {
            *avg /= n;
        }

        // Average group alignments
        let mut avg_groups = [0.0f32; NUM_GROUPS];
        for v in vectors {
            let ga = v.group_alignments.as_array();
            for (avg, &val) in avg_groups.iter_mut().zip(ga.iter()) {
                *avg += val;
            }
        }
        for avg in avg_groups.iter_mut() {
            *avg /= n;
        }

        // Average confidence
        let avg_confidence: f32 = vectors.iter().map(|v| v.confidence).sum::<f32>() / n;

        use crate::teleological::TopicProfile;
        TeleologicalVector::with_all(
            TopicProfile::new(avg_alignments),
            avg_correlations,
            GroupAlignments::from_array(avg_groups),
            avg_confidence,
        )
    }
}
