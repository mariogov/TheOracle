//! Relational Graph Convolutional Network (R-GCN) for memory graph.
//!
//! This module provides inference for the R-GCN model trained via link prediction.
//! The model processes the memory graph with 8 relation types to produce
//! context-aware node embeddings.
//!
//! # Architecture
//!
//! The model is a 2-layer R-GCN:
//! - Layer 1: RGCNLayer(32, 64) + ReLU
//! - Layer 2: RGCNLayer(64, 32)
//!
//! Each layer uses basis decomposition with 4 basis matrices shared across
//! 8 relation types.
//!
//! # Weight Loading
//!
//! Weights are loaded from SafeTensors files exported by the Python training script.
//! See `models/gnn/train_rgcn.py` for training.
//!
//! # Example
//!
//! ```ignore
//! use context_graph_embeddings::models::pretrained::rgcn::RelationalGCN;
//!
//! let rgcn = RelationalGCN::load("models/rgcn/model.safetensors", &device)?;
//!
//! // Build graph from typed edges
//! let edge_index = edges.iter().map(|e| (e.source_idx, e.target_idx)).collect();
//! let edge_types = edges.iter().map(|e| e.edge_type as u8).collect();
//!
//! // Forward pass to get enhanced embeddings
//! let enhanced = rgcn.forward(&node_features, &edge_index, &edge_types)?;
//! ```

mod constants;
mod model;

pub use constants::*;
pub use model::RelationalGCN;
