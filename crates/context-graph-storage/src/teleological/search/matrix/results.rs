//! Matrix search results types.
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**

use uuid::Uuid;

use super::analysis::{CorrelationAnalysis, MatrixAnalysis};
use super::search_matrix::SearchMatrix;
use crate::teleological::search::multi::AggregatedHit;

// ============================================================================
// MATRIX SEARCH RESULTS
// ============================================================================

/// Search results with correlation analysis.
#[derive(Debug, Clone)]
pub struct MatrixSearchResults {
    /// Aggregated hits from underlying multi-search.
    pub hits: Vec<AggregatedHit>,
    /// Correlation analysis between embedders.
    pub correlation: CorrelationAnalysis,
    /// Matrix used for search.
    pub matrix_used: SearchMatrix,
    /// Matrix analysis results.
    pub matrix_analysis: MatrixAnalysis,
    /// Total latency in microseconds.
    pub latency_us: u64,
}

impl MatrixSearchResults {
    /// Check if no results were found.
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    /// Get the top (highest score) result.
    pub fn top(&self) -> Option<&AggregatedHit> {
        self.hits.first()
    }

    /// Get top N results.
    pub fn top_n(&self, n: usize) -> &[AggregatedHit] {
        if n >= self.hits.len() {
            &self.hits
        } else {
            &self.hits[..n]
        }
    }

    /// Get all result IDs.
    pub fn ids(&self) -> Vec<Uuid> {
        self.hits.iter().map(|h| h.id).collect()
    }
}
