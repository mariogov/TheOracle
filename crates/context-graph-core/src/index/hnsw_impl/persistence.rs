//! Persistence operations for RealHnswIndex.
//!
//! Handles loading and saving of HNSW indexes to disk, with validation
//! for legacy format rejection (AP-007).

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, error, info, warn};

use super::real_hnsw::RealHnswIndex;
use super::types::HnswPersistenceData;
use crate::index::error::{IndexError, IndexResult};

impl RealHnswIndex {
    /// Persist the index to disk.
    ///
    /// Saves UUID mappings and vectors. The HNSW graph is rebuilt on load.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the index data
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Successfully persisted
    /// * `Err(IndexError)` - IO or serialization error
    pub fn persist(&self, path: &Path) -> IndexResult<()> {
        // Create persistence data: (UUID, data_id, vector) tuples
        let data: Vec<_> = self
            .uuid_to_data_id()
            .iter()
            .filter_map(|(&uuid, &data_id)| {
                self.stored_vectors()
                    .get(&uuid)
                    .map(|v| (uuid, data_id, v.clone()))
            })
            .collect();

        let file = File::create(path)
            .map_err(|e| IndexError::io("creating RealHnswIndex persistence file", e))?;
        let writer = BufWriter::new(file);

        // Serialize: (config, active_metric, next_data_id, data)
        let persist_data = (
            self.config(),
            self.active_metric(),
            self.next_data_id().load(Ordering::SeqCst),
            data,
        );

        bincode::serialize_into(writer, &persist_data)
            .map_err(|e| IndexError::serialization("serializing RealHnswIndex", e))?;

        debug!(
            "Persisted RealHnswIndex with {} vectors to {:?}",
            self.len(),
            path
        );

        Ok(())
    }

    /// Load the index from disk and rebuild the HNSW graph.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to load the index data from
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - Successfully loaded and rebuilt index
    /// * `Err(IndexError)` - IO, serialization, or construction error
    ///
    /// # Constitution Compliance
    ///
    /// Per AP-007: No backwards compatibility with legacy formats.
    /// This function will reject any legacy SimpleHnswIndex format files.
    pub fn load(path: &Path) -> IndexResult<Self> {
        // AP-007: Reject legacy formats - no backwards compatibility
        // Check file header for legacy SimpleHnswIndex format markers
        let data = std::fs::read(path)
            .map_err(|e| IndexError::io("reading index file for format check", e))?;

        // Check for legacy SimpleHnswIndex format markers
        // These magic bytes were used by the deprecated SimpleHnswIndex serialization
        if data.starts_with(b"SIMPLE_HNSW")
            || data.starts_with(b"\x00SIMPLE")
            || (data.len() > 8 && &data[0..8] == b"SIMP_IDX")
        {
            error!(
                "FATAL: Legacy SimpleHnswIndex format detected at {:?}. \
                 This format was deprecated and is no longer supported.",
                path
            );
            return Err(IndexError::legacy_format(
                path.display().to_string(),
                "Legacy SimpleHnswIndex format detected. \
                 This format was deprecated and is no longer supported. \
                 Data must be reindexed using RealHnswIndex. \
                 See: docs2/codestate/sherlockplans/agent5-backwards-compat-removal.md",
            ));
        }

        // Also reject old .hnsw.bin files based on filename pattern (extra safety)
        if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
            if filename.ends_with(".hnsw.bin") && !filename.contains("real_hnsw") {
                warn!(
                    "Potential legacy index file detected: {:?}. \
                     Attempting to load, but may fail if format is incompatible.",
                    path
                );
            }
        }

        let file = File::open(path)
            .map_err(|e| IndexError::io("opening RealHnswIndex persistence file", e))?;
        let reader = BufReader::new(file);

        // Deserialize: (config, active_metric, next_data_id, data)
        let (config, active_metric, next_data_id, data): HnswPersistenceData =
            bincode::deserialize_from(reader)
                .map_err(|e| IndexError::serialization("deserializing RealHnswIndex", e))?;

        // Create a new index with the loaded config
        let mut index = Self::new(config)?;
        *index.next_data_id_mut() = AtomicUsize::new(next_data_id);
        index.set_active_metric(active_metric);

        // Re-insert all vectors to rebuild the HNSW graph
        for (uuid, _data_id, vector) in data {
            index.add(uuid, &vector)?;
        }

        info!(
            "Loaded RealHnswIndex with {} vectors from {:?}",
            index.len(),
            path
        );

        Ok(index)
    }
}
