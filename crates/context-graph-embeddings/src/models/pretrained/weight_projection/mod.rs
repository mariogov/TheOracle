//! Learned weight projection for graph edge weights.
//!
//! This module provides inference for the CrossEmbedderProjection model trained
//! via contrastive learning. The model projects 13 embedder similarity scores
//! to an optimal edge weight [0, 1].
//!
//! # Architecture
//!
//! The model is a simple 3-layer MLP:
//! - Input: [13] embedder similarity scores
//! - Layer 1: Linear(13, 64) + LayerNorm + GELU
//! - Layer 2: Linear(64, 32) + GELU
//! - Layer 3: Linear(32, 1) + Sigmoid
//!
//! # Weight Loading
//!
//! Weights are loaded from SafeTensors files exported by the Python training script.
//! See `models/graph_weights/train_contrastive.py` for training.
//!
//! # Example
//!
//! ```ignore
//! use context_graph_embeddings::models::pretrained::weight_projection::LearnedWeightProjection;
//!
//! let proj = LearnedWeightProjection::load("models/graph_weights/weights.safetensors")?;
//! let scores = [0.8, 0.0, 0.0, 0.0, 0.7, 0.5, 0.9, 0.6, 0.4, 0.7, 0.8, 0.5, 0.6];
//! let weight = proj.project(&scores)?;
//! assert!(weight >= 0.0 && weight <= 1.0);
//! ```

mod constants;
mod model;

pub use constants::*;
pub use model::LearnedWeightProjection;
