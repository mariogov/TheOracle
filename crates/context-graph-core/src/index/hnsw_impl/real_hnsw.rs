//! RealHnswIndex - Production HNSW implementation using hnsw_rs.
//!
//! # CRITICAL: NO FALLBACKS
//!
//! This is the production HNSW implementation. If construction, insertion,
//! or search fails, detailed errors are returned - no silent degradation.

use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::index::config::{DistanceMetric, EmbedderIndex, HnswConfig};
use crate::index::error::{IndexError, IndexResult};

/// Real HNSW index using hnsw_rs library.
///
/// # CRITICAL: NO FALLBACKS
///
/// This is the production HNSW implementation. If construction, insertion,
/// or search fails, detailed errors are returned - no silent degradation.
///
/// # Thread Safety
///
/// The underlying hnsw_rs::Hnsw is Send + Sync. The wrapper uses interior
/// mutability through the library's internal mechanisms.
///
/// # Persistence
///
/// Vectors are stored alongside UUID mappings for persistence. On load,
/// vectors are re-inserted into a fresh HNSW graph.
pub struct RealHnswIndex {
    /// The actual HNSW index from hnsw_rs (cosine distance)
    inner_cosine: Option<Hnsw<'static, f32, DistCosine>>,
    /// The actual HNSW index from hnsw_rs (L2 distance)
    inner_l2: Option<Hnsw<'static, f32, DistL2>>,
    /// The actual HNSW index from hnsw_rs (dot product)
    inner_dot: Option<Hnsw<'static, f32, DistDot>>,
    /// UUID to data_id mapping (hnsw_rs uses usize internally)
    uuid_to_data_id: HashMap<Uuid, usize>,
    /// Data_id to UUID reverse mapping
    data_id_to_uuid: HashMap<usize, Uuid>,
    /// Stored vectors for persistence (UUID -> vector)
    stored_vectors: HashMap<Uuid, Vec<f32>>,
    /// Next available data_id (atomic for thread-safety during parallel inserts)
    next_data_id: AtomicUsize,
    /// Configuration
    config: HnswConfig,
    /// Whether initialized
    initialized: bool,
    /// Which distance metric is active
    active_metric: DistanceMetric,
}

impl std::fmt::Debug for RealHnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealHnswIndex")
            .field("dimension", &self.config.dimension)
            .field("metric", &self.active_metric)
            .field("num_vectors", &self.uuid_to_data_id.len())
            .field("initialized", &self.initialized)
            .finish()
    }
}

impl RealHnswIndex {
    /// Create a new HNSW index with the given configuration.
    ///
    /// # FAIL FAST
    ///
    /// If HNSW construction fails, returns a detailed error. No fallbacks.
    pub fn new(config: HnswConfig) -> IndexResult<Self> {
        let m = config.m;
        let ef_construction = config.ef_construction;
        let dimension = config.dimension;
        let metric = config.metric;

        info!(
            "Creating RealHnswIndex: dim={}, M={}, ef_construction={}, metric={:?}",
            dimension, m, ef_construction, metric
        );

        // Calculate max_layer based on expected dataset size
        let max_layer = 16;
        let max_elements = 100_000;

        let mut index = Self {
            inner_cosine: None,
            inner_l2: None,
            inner_dot: None,
            uuid_to_data_id: HashMap::new(),
            data_id_to_uuid: HashMap::new(),
            stored_vectors: HashMap::new(),
            next_data_id: AtomicUsize::new(0),
            config: config.clone(),
            initialized: false,
            active_metric: metric,
        };

        // Create the appropriate index based on distance metric
        match metric {
            DistanceMetric::Cosine | DistanceMetric::AsymmetricCosine => {
                let hnsw = Hnsw::<f32, DistCosine>::new(
                    m,
                    max_elements,
                    max_layer,
                    ef_construction,
                    DistCosine {},
                );
                index.inner_cosine = Some(hnsw);
                debug!("Created DistCosine HNSW index");
            }
            DistanceMetric::Euclidean => {
                let hnsw = Hnsw::<f32, DistL2>::new(
                    m,
                    max_elements,
                    max_layer,
                    ef_construction,
                    DistL2 {},
                );
                index.inner_l2 = Some(hnsw);
                debug!("Created DistL2 HNSW index");
            }
            DistanceMetric::DotProduct => {
                let hnsw = Hnsw::<f32, DistDot>::new(
                    m,
                    max_elements,
                    max_layer,
                    ef_construction,
                    DistDot {},
                );
                index.inner_dot = Some(hnsw);
                debug!("Created DistDot HNSW index");
            }
            DistanceMetric::MaxSim => {
                error!("FATAL: MaxSim distance is not compatible with HNSW indexing");
                return Err(IndexError::HnswConstructionFailed {
                    dimension,
                    m,
                    ef_construction,
                    message: "MaxSim distance metric is not compatible with HNSW.".to_string(),
                });
            }
            DistanceMetric::Jaccard => {
                error!("FATAL: Jaccard distance is not compatible with HNSW indexing (use inverted index)");
                return Err(IndexError::HnswConstructionFailed {
                    dimension,
                    m,
                    ef_construction,
                    message: "Jaccard distance metric is for sparse vectors and not compatible with HNSW.".to_string(),
                });
            }
        }

        index.initialized = true;
        info!(
            "RealHnswIndex created successfully: dim={}, metric={:?}",
            dimension, metric
        );
        Ok(index)
    }

    /// Add a vector to the index.
    pub fn add(&mut self, id: Uuid, vector: &[f32]) -> IndexResult<()> {
        if !self.initialized {
            error!("FATAL: Attempted add to uninitialized HNSW index");
            return Err(IndexError::HnswInternalError {
                context: "add".to_string(),
                message: "Index not initialized".to_string(),
            });
        }

        if vector.len() != self.config.dimension {
            error!(
                "FATAL: Dimension mismatch in HNSW add: expected {}, got {}",
                self.config.dimension,
                vector.len()
            );
            return Err(IndexError::DimensionMismatch {
                embedder: EmbedderIndex::E1Semantic,
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            error!("FATAL: Zero-norm vector in HNSW add for memory_id={}", id);
            return Err(IndexError::ZeroNormVector { memory_id: id });
        }

        // STOR-M5: On re-insert, retire old data_id to prevent ghost points.
        // The old vector remains in the HNSW graph but its data_id is removed
        // from data_id_to_uuid, so search results will filter it out.
        if let Some(&old_data_id) = self.uuid_to_data_id.get(&id) {
            warn!(
                "Re-inserting vector for existing UUID {} — retiring old data_id={}",
                id, old_data_id
            );
            self.data_id_to_uuid.remove(&old_data_id);
            self.uuid_to_data_id.remove(&id);
        }

        let data_id = self.next_data_id.fetch_add(1, Ordering::SeqCst);
        self.uuid_to_data_id.insert(id, data_id);
        self.data_id_to_uuid.insert(data_id, id);

        match self.active_metric {
            DistanceMetric::Cosine | DistanceMetric::AsymmetricCosine => {
                if let Some(ref hnsw) = self.inner_cosine {
                    hnsw.insert_slice((vector, data_id));
                    debug!("Inserted vector into DistCosine HNSW: data_id={}", data_id);
                } else {
                    return Err(IndexError::HnswInsertionFailed {
                        memory_id: id,
                        dimension: vector.len(),
                        message: "Cosine HNSW index not initialized".to_string(),
                    });
                }
            }
            DistanceMetric::Euclidean => {
                if let Some(ref hnsw) = self.inner_l2 {
                    hnsw.insert_slice((vector, data_id));
                } else {
                    return Err(IndexError::HnswInsertionFailed {
                        memory_id: id,
                        dimension: vector.len(),
                        message: "L2 HNSW index not initialized".to_string(),
                    });
                }
            }
            DistanceMetric::DotProduct => {
                if let Some(ref hnsw) = self.inner_dot {
                    hnsw.insert_slice((vector, data_id));
                } else {
                    return Err(IndexError::HnswInsertionFailed {
                        memory_id: id,
                        dimension: vector.len(),
                        message: "DotProduct HNSW index not initialized".to_string(),
                    });
                }
            }
            DistanceMetric::MaxSim => {
                return Err(IndexError::HnswInsertionFailed {
                    memory_id: id,
                    dimension: vector.len(),
                    message: "MaxSim is not supported for HNSW".to_string(),
                });
            }
            DistanceMetric::Jaccard => {
                return Err(IndexError::HnswInsertionFailed {
                    memory_id: id,
                    dimension: vector.len(),
                    message: "Jaccard is for sparse vectors and not supported for HNSW".to_string(),
                });
            }
        }

        self.stored_vectors.insert(id, vector.to_vec());
        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> IndexResult<Vec<(Uuid, f32)>> {
        if !self.initialized {
            return Err(IndexError::HnswSearchFailed {
                k,
                query_dim: query.len(),
                message: "Index not initialized".to_string(),
            });
        }

        if query.len() != self.config.dimension {
            return Err(IndexError::DimensionMismatch {
                embedder: EmbedderIndex::E1Semantic,
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        if self.uuid_to_data_id.is_empty() {
            debug!("HNSW search on empty index - returning empty results.");
            return Ok(Vec::new());
        }

        let ef_search = self.config.ef_search.max(k);
        let neighbours: Vec<Neighbour> = match self.active_metric {
            DistanceMetric::Cosine | DistanceMetric::AsymmetricCosine => self
                .inner_cosine
                .as_ref()
                .map(|h| h.search(query, k, ef_search))
                .ok_or_else(|| IndexError::HnswSearchFailed {
                    k,
                    query_dim: query.len(),
                    message: "Cosine HNSW index not available".to_string(),
                })?,
            DistanceMetric::Euclidean => self
                .inner_l2
                .as_ref()
                .map(|h| h.search(query, k, ef_search))
                .ok_or_else(|| IndexError::HnswSearchFailed {
                    k,
                    query_dim: query.len(),
                    message: "L2 HNSW index not available".to_string(),
                })?,
            DistanceMetric::DotProduct => self
                .inner_dot
                .as_ref()
                .map(|h| h.search(query, k, ef_search))
                .ok_or_else(|| IndexError::HnswSearchFailed {
                    k,
                    query_dim: query.len(),
                    message: "DotProduct HNSW index not available".to_string(),
                })?,
            DistanceMetric::MaxSim => {
                return Err(IndexError::HnswSearchFailed {
                    k,
                    query_dim: query.len(),
                    message: "MaxSim is not supported for HNSW".to_string(),
                });
            }
            DistanceMetric::Jaccard => {
                return Err(IndexError::HnswSearchFailed {
                    k,
                    query_dim: query.len(),
                    message: "Jaccard is for sparse vectors and not supported for HNSW".to_string(),
                });
            }
        };

        let results: Vec<(Uuid, f32)> = neighbours
            .into_iter()
            .filter_map(|n| {
                self.data_id_to_uuid.get(&n.d_id).map(|&uuid| {
                    let similarity = match self.active_metric {
                        DistanceMetric::Cosine | DistanceMetric::AsymmetricCosine => {
                            1.0 - n.distance
                        }
                        DistanceMetric::Euclidean => 1.0 / (1.0 + n.distance),
                        DistanceMetric::DotProduct => 1.0 - n.distance,
                        DistanceMetric::MaxSim | DistanceMetric::Jaccard => {
                            unreachable!("MaxSim/Jaccard not supported in HNSW search")
                        }
                    };
                    (uuid, similarity)
                })
            })
            .collect();

        debug!(
            "HNSW search completed: k={}, returned={} results",
            k,
            results.len()
        );
        Ok(results)
    }

    /// Remove a vector from the index (soft delete).
    pub fn remove(&mut self, id: Uuid) -> bool {
        if let Some(&data_id) = self.uuid_to_data_id.get(&id) {
            self.uuid_to_data_id.remove(&id);
            self.data_id_to_uuid.remove(&data_id);
            self.stored_vectors.remove(&id);
            warn!(
                "Removed UUID {} from mappings (data_id={}). Vector remains in graph.",
                id, data_id
            );
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.uuid_to_data_id.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.uuid_to_data_id.is_empty()
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let vector_bytes = self.len() * self.config.dimension * 4;
        let graph_bytes = self.len() * self.config.m * 2 * 8;
        let mapping_bytes = self.len() * (16 + 8 + 8 + 16);
        vector_bytes + graph_bytes + mapping_bytes
    }

    /// Get the number of points in the underlying HNSW graph.
    pub fn hnsw_point_count(&self) -> usize {
        match self.active_metric {
            DistanceMetric::Cosine | DistanceMetric::AsymmetricCosine => self
                .inner_cosine
                .as_ref()
                .map(|h| h.get_nb_point())
                .unwrap_or(0),
            DistanceMetric::Euclidean => self
                .inner_l2
                .as_ref()
                .map(|h| h.get_nb_point())
                .unwrap_or(0),
            DistanceMetric::DotProduct => self
                .inner_dot
                .as_ref()
                .map(|h| h.get_nb_point())
                .unwrap_or(0),
            DistanceMetric::MaxSim | DistanceMetric::Jaccard => 0,
        }
    }

    // === Accessors for persistence module ===

    pub(super) fn uuid_to_data_id(&self) -> &HashMap<Uuid, usize> {
        &self.uuid_to_data_id
    }
    pub(super) fn stored_vectors(&self) -> &HashMap<Uuid, Vec<f32>> {
        &self.stored_vectors
    }
    pub(super) fn config(&self) -> &HnswConfig {
        &self.config
    }
    pub(super) fn active_metric(&self) -> DistanceMetric {
        self.active_metric
    }
    pub(super) fn next_data_id(&self) -> &AtomicUsize {
        &self.next_data_id
    }
    pub(super) fn next_data_id_mut(&mut self) -> &mut AtomicUsize {
        &mut self.next_data_id
    }
    pub(super) fn set_active_metric(&mut self, metric: DistanceMetric) {
        self.active_metric = metric;
    }
}
