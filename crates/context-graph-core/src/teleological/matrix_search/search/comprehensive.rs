//! Comprehensive comparison across all comparison scopes.
//!
//! Provides detailed analysis of teleological vector similarity
//! across multiple dimensions and strategies.

use std::collections::HashMap;

use super::super::super::groups::GroupType;
use super::super::super::types::NUM_EMBEDDERS;
use super::super::super::vector::TeleologicalVector;
use super::super::config::MatrixSearchConfig;
use super::super::types::{ComparisonScope, ComprehensiveComparison, SearchStrategy};
use super::core::TeleologicalMatrixSearch;

impl TeleologicalMatrixSearch {
    /// Compare two vectors across ALL comparison scopes and return comprehensive analysis.
    pub fn comprehensive_comparison(
        &self,
        a: &TeleologicalVector,
        b: &TeleologicalVector,
    ) -> ComprehensiveComparison {
        // Full comparison
        let full = self.similarity_with_breakdown(a, b);

        // Topic profile only
        let tp_config = MatrixSearchConfig {
            scope: ComparisonScope::TopicProfileOnly,
            ..Default::default()
        };
        let tp_search = TeleologicalMatrixSearch::with_config(tp_config);
        let topic_profile_only = tp_search.similarity(a, b);

        // Cross-correlations only
        let cc_config = MatrixSearchConfig {
            scope: ComparisonScope::CrossCorrelationsOnly,
            ..Default::default()
        };
        let cc_search = TeleologicalMatrixSearch::with_config(cc_config);
        let correlations_only = cc_search.similarity(a, b);

        // Group alignments only
        let ga_config = MatrixSearchConfig {
            scope: ComparisonScope::GroupAlignmentsOnly,
            ..Default::default()
        };
        let ga_search = TeleologicalMatrixSearch::with_config(ga_config);
        let groups_only = ga_search.similarity(a, b);

        // Per-group comparisons
        let mut per_group = HashMap::new();
        for group in GroupType::ALL {
            let group_config = MatrixSearchConfig {
                scope: ComparisonScope::SpecificGroups(vec![group]),
                ..Default::default()
            };
            let group_search = TeleologicalMatrixSearch::with_config(group_config);
            per_group.insert(group, group_search.similarity(a, b));
        }

        // Per-embedder pattern comparisons
        let mut per_embedder_pattern = [0.0; NUM_EMBEDDERS];
        for (embedder, pattern) in per_embedder_pattern.iter_mut().enumerate() {
            let emb_config = MatrixSearchConfig {
                scope: ComparisonScope::SingleEmbedderPattern(embedder),
                ..Default::default()
            };
            let emb_search = TeleologicalMatrixSearch::with_config(emb_config);
            *pattern = emb_search.similarity(a, b);
        }

        // Tucker comparison (if available)
        let tucker = if a.has_tucker_core() && b.has_tucker_core() {
            let tucker_config = MatrixSearchConfig {
                strategy: SearchStrategy::TuckerCompressed,
                ..Default::default()
            };
            let tucker_search = TeleologicalMatrixSearch::with_config(tucker_config);
            Some(tucker_search.similarity(a, b))
        } else {
            None
        };

        ComprehensiveComparison {
            full,
            topic_profile_only,
            correlations_only,
            groups_only,
            per_group,
            per_embedder_pattern,
            tucker,
        }
    }
}
