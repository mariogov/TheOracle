//! HnswMultiSpaceIndex struct and helper methods.
//!
//! The trait implementation is in `multi_space_trait.rs`.

use std::collections::HashMap;
use std::path::Path;

use super::real_hnsw::RealHnswIndex;
use crate::index::config::{DistanceMetric, EmbedderIndex, HnswConfig};
use crate::index::error::IndexResult;
use crate::index::splade_impl::SpladeInvertedIndex;
use crate::index::status::IndexStatus;
use uuid::Uuid;

/// HnswMultiSpaceIndex manages 12 HNSW indexes + SPLADE inverted index.
///
/// # CRITICAL: Uses Real HNSW Implementation
///
/// This implementation uses the real hnsw_rs library for O(log n) approximate
/// nearest neighbor search. NO FALLBACKS - if HNSW operations fail, errors
/// are propagated with full context.
///
/// # Architecture
///
/// - 10 dense HNSW indexes (E1-E5, E7-E11)
/// - 1 Matryoshka 128D HNSW (E1 truncated for Stage 2)
/// - 1 SPLADE inverted index (Stage 1)
///
/// # Thread Safety
///
/// The struct is Send + Sync through interior mutability patterns.
/// The underlying hnsw_rs indexes are thread-safe.
#[derive(Debug)]
pub struct HnswMultiSpaceIndex {
    /// Map from EmbedderIndex to real HNSW index
    hnsw_indexes: HashMap<EmbedderIndex, RealHnswIndex>,
    /// SPLADE inverted index for Stage 1
    splade_index: SpladeInvertedIndex,
    /// Whether initialized
    initialized: bool,
    /// Track HNSW configs for status reporting
    configs: HashMap<EmbedderIndex, HnswConfig>,
}

impl HnswMultiSpaceIndex {
    /// Create a new uninitialized multi-space index.
    pub fn new() -> Self {
        Self {
            hnsw_indexes: HashMap::new(),
            splade_index: SpladeInvertedIndex::new(),
            initialized: false,
            configs: HashMap::new(),
        }
    }

    /// Create HNSW config for a given embedder.
    pub(super) fn config_for_embedder(embedder: EmbedderIndex) -> Option<HnswConfig> {
        let dim = embedder.dimension()?;
        let metric = embedder
            .recommended_metric()
            .unwrap_or(DistanceMetric::Cosine);

        if embedder == EmbedderIndex::E1Matryoshka128 {
            Some(HnswConfig::matryoshka_128d())
        } else {
            Some(HnswConfig::default_for_dimension(dim, metric))
        }
    }

    /// Get index status for a specific embedder.
    pub(super) fn get_embedder_status(&self, embedder: EmbedderIndex) -> IndexStatus {
        if let Some(index) = self.hnsw_indexes.get(&embedder) {
            let mut status = IndexStatus::new_empty(embedder);
            let bytes_per_element = self
                .configs
                .get(&embedder)
                .map(|c| c.estimated_memory_per_vector())
                .unwrap_or(4096);
            status.update_count(index.len(), bytes_per_element);
            return status;
        }

        IndexStatus::uninitialized(embedder)
    }

    // === Accessors for trait implementation ===

    pub(super) fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub(super) fn set_initialized(&mut self, val: bool) {
        self.initialized = val;
    }

    pub(super) fn hnsw_indexes(&self) -> &HashMap<EmbedderIndex, RealHnswIndex> {
        &self.hnsw_indexes
    }

    pub(super) fn hnsw_indexes_mut(&mut self) -> &mut HashMap<EmbedderIndex, RealHnswIndex> {
        &mut self.hnsw_indexes
    }

    pub(super) fn get_hnsw_index(&self, embedder: &EmbedderIndex) -> Option<&RealHnswIndex> {
        self.hnsw_indexes.get(embedder)
    }

    pub(super) fn get_hnsw_index_mut(
        &mut self,
        embedder: &EmbedderIndex,
    ) -> Option<&mut RealHnswIndex> {
        self.hnsw_indexes.get_mut(embedder)
    }

    pub(super) fn insert_hnsw_index(&mut self, embedder: EmbedderIndex, index: RealHnswIndex) {
        self.hnsw_indexes.insert(embedder, index);
    }

    pub(super) fn insert_config(&mut self, embedder: EmbedderIndex, config: HnswConfig) {
        self.configs.insert(embedder, config);
    }

    pub(super) fn hnsw_count(&self) -> usize {
        self.hnsw_indexes.len()
    }

    // === SPLADE index accessors ===

    pub(super) fn add_splade_internal(
        &mut self,
        memory_id: Uuid,
        sparse: &[(usize, f32)],
    ) -> IndexResult<()> {
        self.splade_index.add(memory_id, sparse)
    }

    pub(super) fn search_splade_internal(
        &self,
        sparse_query: &[(usize, f32)],
        k: usize,
    ) -> Vec<(Uuid, f32)> {
        self.splade_index.search(sparse_query, k)
    }

    pub(super) fn remove_splade(&mut self, memory_id: Uuid) -> bool {
        self.splade_index.remove(memory_id)
    }

    pub(super) fn splade_len(&self) -> usize {
        self.splade_index.len()
    }

    pub(super) fn persist_splade(&self, path: &Path) -> IndexResult<()> {
        self.splade_index.persist(path)
    }

    pub(super) fn load_splade(&mut self, path: &Path) -> IndexResult<()> {
        self.splade_index = SpladeInvertedIndex::load(path)?;
        Ok(())
    }
}

impl Default for HnswMultiSpaceIndex {
    fn default() -> Self {
        Self::new()
    }
}
