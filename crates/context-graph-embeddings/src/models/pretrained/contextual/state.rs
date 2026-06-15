//! Model state for ContextualModel.
//!
//! Manages the loaded/unloaded state of the contextual embedding model.
//!
//! # E5-base-v2 Architecture
//!
//! E5-base-v2 uses prefix-based asymmetric encoding, so no projection
//! weights are needed. The model handles asymmetry through:
//! - "query: " prefix for intent embeddings
//! - "passage: " prefix for context embeddings

use crate::gpu::BertWeights;

pub(crate) use crate::models::pretrained::shared::ModelState;

/// Concrete state type for contextual model (BERT weights).
pub type ContextualModelState = ModelState<Box<BertWeights>>;
