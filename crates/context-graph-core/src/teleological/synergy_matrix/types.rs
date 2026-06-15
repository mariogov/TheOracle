//! Type definitions for the synergy matrix.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::constants::SYNERGY_DIM;

/// 13x13 cross-embedding synergy matrix per teleoplan.md.
///
/// Captures the strength of relationships between different embedding spaces.
/// Used for weighting cross-correlations in teleological fusion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynergyMatrix {
    /// 13x13 matrix of synergy values, symmetric with diagonal = 1.0
    pub values: [[f32; SYNERGY_DIM]; SYNERGY_DIM],
    /// Per-cell weights for adaptive learning
    pub weights: [[f32; SYNERGY_DIM]; SYNERGY_DIM],
    /// When this matrix was computed/updated
    pub computed_at: DateTime<Utc>,
    /// Number of samples used to compute/refine the matrix
    pub sample_count: u64,
}
