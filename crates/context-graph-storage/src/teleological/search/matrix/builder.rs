//! MatrixSearchBuilder: fluent API for matrix strategy search.
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**

use std::collections::HashMap;

use super::results::MatrixSearchResults;
use super::search_matrix::SearchMatrix;
use super::strategy_search::MatrixStrategySearch;
use crate::teleological::indexes::EmbedderIndex;
use crate::teleological::search::error::SearchResult;

// ============================================================================
// BUILDER
// ============================================================================

/// Builder pattern for matrix strategy search.
pub struct MatrixSearchBuilder {
    queries: HashMap<EmbedderIndex, Vec<f32>>,
    matrix: SearchMatrix,
    k: usize,
    threshold: Option<f32>,
}

impl MatrixSearchBuilder {
    /// Create a new builder with queries.
    pub fn new(queries: HashMap<EmbedderIndex, Vec<f32>>) -> Self {
        Self {
            queries,
            matrix: SearchMatrix::default(),
            k: 100,
            threshold: None,
        }
    }

    /// Set the search matrix.
    pub fn matrix(mut self, matrix: SearchMatrix) -> Self {
        self.matrix = matrix;
        self
    }

    /// Set the number of results to return.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set minimum similarity threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Execute the search.
    pub fn execute(self, search: &MatrixStrategySearch) -> SearchResult<MatrixSearchResults> {
        search.search(self.queries, self.matrix, self.k, self.threshold)
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        println!("=== TEST: MatrixSearchBuilder pattern ===");

        let queries: HashMap<EmbedderIndex, Vec<f32>> = HashMap::new();
        let builder = MatrixSearchBuilder::new(queries.clone())
            .matrix(SearchMatrix::code_heavy())
            .k(50)
            .threshold(0.5);

        assert_eq!(builder.k, 50);
        assert_eq!(builder.threshold, Some(0.5));
        assert_eq!(builder.matrix, SearchMatrix::code_heavy());

        println!("RESULT: PASS");
    }
}
