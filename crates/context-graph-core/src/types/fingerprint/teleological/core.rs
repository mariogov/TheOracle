//! Core implementation for TeleologicalFingerprint.
//!
//! This module contains constructors, constants, and core methods.

use chrono::Utc;
use uuid::Uuid;

use crate::types::fingerprint::{SemanticFingerprint, SparseVector};

use super::types::TeleologicalFingerprint;

impl TeleologicalFingerprint {
    /// Expected size in bytes for a complete teleological fingerprint.
    /// From constitution.yaml: ~46KB per node (+ ~2KB if e6_sparse present).
    pub const EXPECTED_SIZE_BYTES: usize = 46_000;

    /// Additional size for E6 sparse vector (~235 terms * 6 bytes).
    pub const E6_SPARSE_SIZE_BYTES: usize = 1_500;

    /// Default importance score for new fingerprints.
    pub const DEFAULT_IMPORTANCE: f32 = 0.5;

    /// Create a new TeleologicalFingerprint with default importance (0.5).
    ///
    /// Automatically:
    /// - Generates a new UUID v4
    /// - Sets timestamps to now
    /// - Sets importance to 0.5 (default)
    /// - Sets e6_sparse to None (use with_e6_sparse to set)
    ///
    /// # Arguments
    /// * `semantic` - The semantic fingerprint (13 embeddings)
    /// * `content_hash` - SHA-256 hash of source content
    pub fn new(semantic: SemanticFingerprint, content_hash: [u8; 32]) -> Self {
        Self::with_importance(semantic, content_hash, Self::DEFAULT_IMPORTANCE)
    }

    /// Create a new TeleologicalFingerprint with specific importance.
    ///
    /// # Arguments
    /// * `semantic` - The semantic fingerprint (13 embeddings)
    /// * `content_hash` - SHA-256 hash of source content
    /// * `importance` - Importance score [0.0, 1.0], clamped if out of range
    pub fn with_importance(
        semantic: SemanticFingerprint,
        content_hash: [u8; 32],
        importance: f32,
    ) -> Self {
        let now = Utc::now();

        Self {
            id: Uuid::new_v4(),
            semantic,
            content_hash,
            created_at: now,
            last_updated: now,
            access_count: 0,
            importance: importance.clamp(0.0, 1.0),
            last_accessed_at: now,
            e6_sparse: None,
        }
    }

    /// Create a TeleologicalFingerprint with a specific ID (for testing/import).
    pub fn with_id(id: Uuid, semantic: SemanticFingerprint, content_hash: [u8; 32]) -> Self {
        let mut fp = Self::new(semantic, content_hash);
        fp.id = id;
        fp
    }

    /// Builder pattern: set the E6 sparse vector.
    ///
    /// # Arguments
    /// * `sparse` - The original E6 sparse vector (before projection)
    ///
    /// # Example
    /// ```ignore
    /// let fp = TeleologicalFingerprint::new(semantic, hash)
    ///     .with_e6_sparse(sparse_vec);
    /// ```
    pub fn with_e6_sparse(mut self, sparse: SparseVector) -> Self {
        self.e6_sparse = Some(sparse);
        self
    }

    /// Set the E6 sparse vector (mutable reference version).
    ///
    /// Use this when you need to update an existing fingerprint.
    pub fn set_e6_sparse(&mut self, sparse: SparseVector) {
        self.e6_sparse = Some(sparse);
        self.last_updated = Utc::now();
    }

    /// Check if this fingerprint has an E6 sparse vector.
    #[inline]
    pub fn has_e6_sparse(&self) -> bool {
        self.e6_sparse.is_some()
    }

    /// Get the E6 sparse vector, if present.
    #[inline]
    pub fn e6_sparse(&self) -> Option<&SparseVector> {
        self.e6_sparse.as_ref()
    }

    /// Compute E6 term overlap score with a query sparse vector.
    ///
    /// Returns the fraction of query terms that appear in this document.
    /// Used for tie-breaking when E1 scores are similar.
    ///
    /// # Returns
    /// - `Some(score)` where score is in [0.0, 1.0] if e6_sparse is present
    /// - `None` if no e6_sparse vector is stored
    pub fn e6_term_overlap(&self, query_sparse: &SparseVector) -> Option<f32> {
        self.e6_sparse.as_ref().map(|doc_sparse| {
            if query_sparse.nnz() == 0 {
                return 0.0;
            }
            // Count shared terms using merge-join
            let mut shared = 0usize;
            let mut i = 0;
            let mut j = 0;
            while i < query_sparse.indices.len() && j < doc_sparse.indices.len() {
                match query_sparse.indices[i].cmp(&doc_sparse.indices[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        shared += 1;
                        i += 1;
                        j += 1;
                    }
                }
            }
            shared as f32 / query_sparse.nnz() as f32
        })
    }

    /// Record an access event.
    ///
    /// Increments access_count and updates last_updated and last_accessed_at timestamps.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        let now = Utc::now();
        self.last_updated = now;
        self.last_accessed_at = now;
    }

    /// Get the age of this fingerprint (time since creation).
    pub fn age(&self) -> chrono::Duration {
        Utc::now() - self.created_at
    }

    /// Get the total memory size of this fingerprint including E6 sparse.
    pub fn total_size(&self) -> usize {
        let base_size = Self::EXPECTED_SIZE_BYTES;
        let sparse_size = self
            .e6_sparse
            .as_ref()
            .map(|s| s.memory_size())
            .unwrap_or(0);
        base_size + sparse_size
    }
}
