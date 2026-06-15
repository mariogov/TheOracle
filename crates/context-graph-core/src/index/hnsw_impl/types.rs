//! Type definitions for HNSW index implementation.
//!
//! Contains type aliases and data structures used for persistence.

use uuid::Uuid;

use super::HnswConfig;
use crate::index::config::DistanceMetric;

/// Type alias for HNSW persistence data to reduce type complexity.
///
/// Contains:
/// - `HnswConfig` - Index configuration
/// - `DistanceMetric` - Active distance metric
/// - `usize` - Next data ID counter
/// - `Vec<(Uuid, usize, Vec<f32>)>` - Vector data: (UUID, data_id, vector)
pub type HnswPersistenceData = (
    HnswConfig,
    DistanceMetric,
    usize,
    Vec<(Uuid, usize, Vec<f32>)>,
);
