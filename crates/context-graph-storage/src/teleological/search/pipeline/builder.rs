//! Pipeline builder pattern for constructing queries.
//!
//! Provides a fluent API for building and executing pipeline queries.

use tracing::error;

use super::execution::RetrievalPipeline;
use super::types::{PipelineError, PipelineResult, PipelineStage};

// ============================================================================
// PIPELINE BUILDER
// ============================================================================

/// Builder for pipeline queries.
pub struct PipelineBuilder {
    pub(crate) query_splade: Option<Vec<(usize, f32)>>,
    pub(crate) query_matryoshka: Option<Vec<f32>>,
    pub(crate) query_semantic: Option<Vec<f32>>,
    pub(crate) query_tokens: Option<Vec<Vec<f32>>>,
    pub(crate) stages: Option<Vec<PipelineStage>>,
    pub(crate) k: Option<usize>,
}

impl PipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            query_splade: None,
            query_matryoshka: None,
            query_semantic: None,
            query_tokens: None,
            stages: None,
            k: None,
        }
    }

    /// Set SPLADE query (sparse vector as term_id, weight pairs).
    pub fn splade(mut self, query: Vec<(usize, f32)>) -> Self {
        self.query_splade = Some(query);
        self
    }

    /// Set Matryoshka 128D query.
    pub fn matryoshka(mut self, query: Vec<f32>) -> Self {
        self.query_matryoshka = Some(query);
        self
    }

    /// Set semantic 1024D query.
    pub fn semantic(mut self, query: Vec<f32>) -> Self {
        self.query_semantic = Some(query);
        self
    }

    /// Set token embeddings for MaxSim (each 128D).
    pub fn tokens(mut self, query: Vec<Vec<f32>>) -> Self {
        self.query_tokens = Some(query);
        self
    }

    /// Set stages to execute.
    pub fn stages(mut self, stages: Vec<PipelineStage>) -> Self {
        self.stages = Some(stages);
        self
    }

    /// Set final result limit.
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Execute the pipeline.
    pub fn execute(self, pipeline: &RetrievalPipeline) -> Result<PipelineResult, PipelineError> {
        let query_splade = self.query_splade.unwrap_or_default();
        // When no matryoshka query is provided, skip MatryoshkaAnn stage instead of
        // defaulting to zero vector (which would pollute ANN results with nonsense scores).
        let has_matryoshka = self.query_matryoshka.is_some();
        let query_matryoshka = self.query_matryoshka.unwrap_or_default();
        let query_semantic = self.query_semantic.ok_or_else(|| {
            error!("Pipeline build failed: query_semantic (E1) is required but not set");
            PipelineError::MissingQuery {
                stage: PipelineStage::RrfRerank,
            }
        })?;
        let query_tokens = self.query_tokens.unwrap_or_default();

        let mut stages = self.stages.unwrap_or_else(|| PipelineStage::all().to_vec());
        if !has_matryoshka {
            stages.retain(|s| *s != PipelineStage::MatryoshkaAnn);
        }

        let requested_k = self.k;

        let mut result = pipeline.execute_stages(
            &query_splade,
            &query_matryoshka,
            &query_semantic,
            &query_tokens,
            &stages,
        )?;

        // H2 FIX (Audit #10): Apply builder's k as post-execution truncation.
        // execute_stages uses pipeline.config.k internally for stage-level limits.
        // The builder's k controls the FINAL result count returned to the caller.
        // Previously, the builder computed a modified config but never passed it,
        // silently ignoring the caller's .k() value.
        if let Some(k) = requested_k {
            result.results.truncate(k);
        }

        Ok(result)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
