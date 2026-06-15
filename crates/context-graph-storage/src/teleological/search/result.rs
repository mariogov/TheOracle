//! Search result types for single embedder search.
//!
//! # Types
//!
//! - [`EmbedderSearchHit`]: A single search result with ID, distance, similarity
//! - [`SingleEmbedderSearchResults`]: Collection of hits from one embedder
//!
//! # Distance to Similarity Conversion
//!
//! HNSW returns **distance** (lower = more similar). We convert to **similarity**
//! for consistency with `compute_cosine_similarity()`:
//!
//! ```text
//! Cosine distance ∈ [0, 2]  →  similarity = (2 - distance) / 2  →  ∈ [0, 1]
//! STOR-10 FIX: Normalized to match direct cosine computation: (cos+1)/2
//! ```

use uuid::Uuid;

use super::super::indexes::EmbedderIndex;

/// A single search result from an embedder index.
///
/// Contains both the raw distance from HNSW and the converted similarity score.
///
/// # Fields
///
/// - `id`: Memory UUID
/// - `distance`: Raw HNSW distance (lower = more similar)
/// - `similarity`: Converted score [0.0, 1.0] (higher = more similar)
/// - `embedder`: Which embedder was searched
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::EmbedderSearchHit;
/// use context_graph_storage::teleological::indexes::EmbedderIndex;
/// use uuid::Uuid;
///
/// let hit = EmbedderSearchHit::from_hnsw(
///     Uuid::new_v4(),
///     0.1,  // 10% distance → (2-0.1)/2 = 0.95 similarity
///     EmbedderIndex::E1Semantic,
/// );
///
/// assert!(hit.similarity > 0.94 && hit.similarity < 0.96);
/// ```
#[derive(Debug, Clone)]
pub struct EmbedderSearchHit {
    /// The memory ID (fingerprint UUID).
    pub id: Uuid,

    /// Distance from query (lower = more similar for HNSW).
    ///
    /// Note: HNSW returns distance, not similarity.
    /// For cosine metric: distance ∈ [0, 2] where 0 = identical.
    pub distance: f32,

    /// Similarity score [0.0, 1.0] (converted from distance).
    ///
    /// STOR-10 FIX: similarity = (2.0 - distance) / 2.0, matching compute_cosine_similarity().
    pub similarity: f32,

    /// Which embedder was searched.
    pub embedder: EmbedderIndex,
}

impl EmbedderSearchHit {
    /// Create from HNSW search result (id, distance).
    ///
    /// Converts distance to similarity using normalized formula:
    /// - distance 0.0 → similarity 1.0 (identical)
    /// - distance 1.0 → similarity 0.5 (orthogonal)
    /// - distance 2.0 → similarity 0.0 (opposite)
    ///
    /// STOR-10 FIX: Uses `(2 - distance) / 2` to match `compute_cosine_similarity()`.
    ///
    /// # Arguments
    ///
    /// * `id` - Memory UUID
    /// * `distance` - HNSW distance (typically cosine distance)
    /// * `embedder` - Which embedder was searched
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_storage::teleological::search::EmbedderSearchHit;
    /// use context_graph_storage::teleological::indexes::EmbedderIndex;
    /// use uuid::Uuid;
    ///
    /// // Identical vectors (distance 0)
    /// let hit = EmbedderSearchHit::from_hnsw(
    ///     Uuid::new_v4(),
    ///     0.0,
    ///     EmbedderIndex::E8Graph,
    /// );
    /// assert!((hit.similarity - 1.0).abs() < 0.001);
    ///
    /// // Orthogonal vectors (distance 1) — now 0.5, not 0.0
    /// let hit = EmbedderSearchHit::from_hnsw(
    ///     Uuid::new_v4(),
    ///     1.0,
    ///     EmbedderIndex::E8Graph,
    /// );
    /// assert!((hit.similarity - 0.5).abs() < 0.001);
    /// ```
    #[inline]
    pub fn from_hnsw(id: Uuid, distance: f32, embedder: EmbedderIndex) -> Self {
        // STOR-10 FIX: Normalize to match compute_cosine_similarity(): (cos+1)/2
        let similarity = ((2.0 - distance) / 2.0).clamp(0.0, 1.0);
        Self {
            id,
            distance,
            similarity,
            embedder,
        }
    }

    /// Check if this hit has high similarity (>= 0.9).
    #[inline]
    pub fn is_high_similarity(&self) -> bool {
        self.similarity >= 0.9
    }

    /// Check if this hit meets a minimum similarity threshold.
    #[inline]
    pub fn meets_threshold(&self, min_similarity: f32) -> bool {
        self.similarity >= min_similarity
    }
}

/// Results from a single embedder search.
///
/// Contains hits sorted by similarity descending, plus metadata about the search.
///
/// # Fields
///
/// - `hits`: Search results sorted by similarity (highest first)
/// - `embedder`: Which embedder was searched
/// - `k`: Requested limit
/// - `threshold`: Minimum similarity filter (if applied)
/// - `latency_us`: Search latency in microseconds
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::{
///     EmbedderSearchHit, SingleEmbedderSearchResults,
/// };
/// use context_graph_storage::teleological::indexes::EmbedderIndex;
/// use uuid::Uuid;
///
/// let results = SingleEmbedderSearchResults {
///     hits: vec![
///         EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.1, EmbedderIndex::E1Semantic),
///         EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.3, EmbedderIndex::E1Semantic),
///     ],
///     embedder: EmbedderIndex::E1Semantic,
///     k: 10,
///     threshold: Some(0.5),
///     latency_us: 150,
/// };
///
/// assert_eq!(results.len(), 2);
/// assert!(results.top().unwrap().similarity > 0.8);
/// ```
#[derive(Debug, Clone)]
pub struct SingleEmbedderSearchResults {
    /// Hits sorted by similarity descending.
    pub hits: Vec<EmbedderSearchHit>,

    /// Which embedder was searched.
    pub embedder: EmbedderIndex,

    /// Query k (requested limit).
    pub k: usize,

    /// Threshold applied (if any).
    pub threshold: Option<f32>,

    /// Search latency in microseconds.
    pub latency_us: u64,
}

impl SingleEmbedderSearchResults {
    /// Check if no results were found.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Get the number of results.
    #[inline]
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    /// Get the top (most similar) result.
    #[inline]
    pub fn top(&self) -> Option<&EmbedderSearchHit> {
        self.hits.first()
    }

    /// Get all result IDs.
    #[inline]
    pub fn ids(&self) -> Vec<Uuid> {
        self.hits.iter().map(|h| h.id).collect()
    }

    /// Get top N results.
    #[inline]
    pub fn top_n(&self, n: usize) -> &[EmbedderSearchHit] {
        if n >= self.hits.len() {
            &self.hits
        } else {
            &self.hits[..n]
        }
    }

    /// Get results with similarity above threshold.
    #[inline]
    pub fn above_threshold(&self, min_similarity: f32) -> Vec<&EmbedderSearchHit> {
        self.hits
            .iter()
            .filter(|h| h.similarity >= min_similarity)
            .collect()
    }

    /// Get average similarity of all hits.
    #[inline]
    pub fn average_similarity(&self) -> Option<f32> {
        if self.hits.is_empty() {
            None
        } else {
            let sum: f32 = self.hits.iter().map(|h| h.similarity).sum();
            Some(sum / self.hits.len() as f32)
        }
    }

    /// Get iterator over hits.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &EmbedderSearchHit> {
        self.hits.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_from_hnsw_identical_vectors() {
        println!("=== TEST: Identical vectors have distance 0, similarity 1 ===");
        println!("BEFORE: distance = 0.0");

        let hit = EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.0, EmbedderIndex::E1Semantic);

        println!("AFTER: similarity = {}", hit.similarity);
        assert!((hit.similarity - 1.0).abs() < 1e-6);
        assert!(hit.is_high_similarity());

        println!("RESULT: PASS");
    }

    #[test]
    fn test_hit_from_hnsw_orthogonal_vectors() {
        println!("=== TEST: Orthogonal vectors have distance 1, similarity 0.5 (STOR-10) ===");
        println!("BEFORE: distance = 1.0");

        let hit = EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 1.0, EmbedderIndex::E1Semantic);

        println!("AFTER: similarity = {}", hit.similarity);
        // STOR-10: (2.0 - 1.0) / 2.0 = 0.5 — orthogonal is mid-range, not zero
        assert!((hit.similarity - 0.5).abs() < 1e-6);
        assert!(!hit.is_high_similarity()); // 0.5 < 0.9

        println!("RESULT: PASS");
    }

    #[test]
    fn test_hit_from_hnsw_opposite_vectors() {
        println!("=== TEST: Opposite vectors have distance 2, similarity clamped to 0 ===");
        println!("BEFORE: distance = 2.0");

        let hit = EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 2.0, EmbedderIndex::E1Semantic);

        println!("AFTER: similarity = {}", hit.similarity);
        assert_eq!(hit.similarity, 0.0); // Clamped

        println!("RESULT: PASS");
    }

    #[test]
    fn test_hit_from_hnsw_partial_similarity() {
        println!("=== TEST: Partial similarity (distance 0.3) ===");
        println!("BEFORE: distance = 0.3");

        let hit = EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.3, EmbedderIndex::E8Graph);

        println!("AFTER: similarity = {}", hit.similarity);
        // STOR-10: (2.0 - 0.3) / 2.0 = 0.85
        assert!((hit.similarity - 0.85).abs() < 1e-6);
        assert!(!hit.is_high_similarity()); // 0.85 < 0.9

        println!("RESULT: PASS");
    }

    #[test]
    fn test_hit_meets_threshold() {
        println!("=== TEST: meets_threshold() ===");

        // STOR-10: distance 0.2 → similarity = (2.0 - 0.2) / 2.0 = 0.9
        let hit = EmbedderSearchHit::from_hnsw(
            Uuid::new_v4(),
            0.2, // similarity = 0.9
            EmbedderIndex::E1Semantic,
        );

        assert!(hit.meets_threshold(0.5));
        assert!(hit.meets_threshold(0.8));
        assert!(hit.meets_threshold(0.9));
        assert!(!hit.meets_threshold(0.91));

        println!("RESULT: PASS");
    }

    #[test]
    fn test_results_empty() {
        println!("=== TEST: Empty results ===");

        let results = SingleEmbedderSearchResults {
            hits: vec![],
            embedder: EmbedderIndex::E1Semantic,
            k: 10,
            threshold: None,
            latency_us: 100,
        };

        assert!(results.is_empty());
        assert_eq!(results.len(), 0);
        assert!(results.top().is_none());
        assert!(results.ids().is_empty());
        assert!(results.average_similarity().is_none());

        println!("RESULT: PASS");
    }

    #[test]
    fn test_results_with_hits() {
        println!("=== TEST: Results with multiple hits ===");

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        // STOR-10: distances 0.1/0.3/0.5 → sims 0.95/0.85/0.75
        let results = SingleEmbedderSearchResults {
            hits: vec![
                EmbedderSearchHit::from_hnsw(id1, 0.1, EmbedderIndex::E1Semantic), // sim 0.95
                EmbedderSearchHit::from_hnsw(id2, 0.3, EmbedderIndex::E1Semantic), // sim 0.85
                EmbedderSearchHit::from_hnsw(id3, 0.5, EmbedderIndex::E1Semantic), // sim 0.75
            ],
            embedder: EmbedderIndex::E1Semantic,
            k: 10,
            threshold: None,
            latency_us: 250,
        };

        assert!(!results.is_empty());
        assert_eq!(results.len(), 3);

        let top = results.top().unwrap();
        assert_eq!(top.id, id1);
        assert!((top.similarity - 0.95).abs() < 1e-6);

        let ids = results.ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&id1));

        println!("RESULT: PASS");
    }

    #[test]
    fn test_results_top_n() {
        println!("=== TEST: top_n() ===");

        let results = SingleEmbedderSearchResults {
            hits: vec![
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.1, EmbedderIndex::E1Semantic),
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.2, EmbedderIndex::E1Semantic),
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.3, EmbedderIndex::E1Semantic),
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.4, EmbedderIndex::E1Semantic),
            ],
            embedder: EmbedderIndex::E1Semantic,
            k: 10,
            threshold: None,
            latency_us: 100,
        };

        assert_eq!(results.top_n(2).len(), 2);
        assert_eq!(results.top_n(10).len(), 4); // Only 4 available
        assert_eq!(results.top_n(0).len(), 0);

        println!("RESULT: PASS");
    }

    #[test]
    fn test_results_above_threshold() {
        println!("=== TEST: above_threshold() ===");

        // STOR-10: distances 0.1/0.3/0.6 → sims 0.95/0.85/0.70
        let results = SingleEmbedderSearchResults {
            hits: vec![
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.1, EmbedderIndex::E1Semantic), // 0.95
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.3, EmbedderIndex::E1Semantic), // 0.85
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.6, EmbedderIndex::E1Semantic), // 0.70
            ],
            embedder: EmbedderIndex::E1Semantic,
            k: 10,
            threshold: None,
            latency_us: 100,
        };

        let above_90 = results.above_threshold(0.9);
        assert_eq!(above_90.len(), 1); // only 0.95

        let above_80 = results.above_threshold(0.8);
        assert_eq!(above_80.len(), 2); // 0.95 and 0.85

        let above_60 = results.above_threshold(0.6);
        assert_eq!(above_60.len(), 3); // all three

        println!("RESULT: PASS");
    }

    #[test]
    fn test_results_average_similarity() {
        println!("=== TEST: average_similarity() ===");

        // STOR-10: distances 0.0/0.5/1.0 → sims 1.0/0.75/0.5
        let results = SingleEmbedderSearchResults {
            hits: vec![
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.0, EmbedderIndex::E1Semantic), // 1.0
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.5, EmbedderIndex::E1Semantic), // 0.75
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 1.0, EmbedderIndex::E1Semantic), // 0.5
            ],
            embedder: EmbedderIndex::E1Semantic,
            k: 10,
            threshold: None,
            latency_us: 100,
        };

        let avg = results.average_similarity().unwrap();
        println!("Average similarity: {}", avg);
        // (1.0 + 0.75 + 0.5) / 3 = 0.75
        assert!((avg - 0.75).abs() < 1e-6);

        println!("RESULT: PASS");
    }

    #[test]
    fn test_results_iter() {
        println!("=== TEST: iter() ===");

        let results = SingleEmbedderSearchResults {
            hits: vec![
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.1, EmbedderIndex::E1Semantic),
                EmbedderSearchHit::from_hnsw(Uuid::new_v4(), 0.2, EmbedderIndex::E1Semantic),
            ],
            embedder: EmbedderIndex::E1Semantic,
            k: 10,
            threshold: None,
            latency_us: 100,
        };

        let count = results.iter().count();
        assert_eq!(count, 2);

        for hit in results.iter() {
            assert!(hit.similarity > 0.0);
        }

        println!("RESULT: PASS");
    }
}
