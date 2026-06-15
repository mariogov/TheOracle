//! Model loading utilities for the sparse SPLADE model.
//!
//! This module handles loading MLM head weights from safetensors files.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::gpu::BertConfig;

use super::types::MlmHeadWeights;

/// Load MLM head weights from safetensors.
pub(crate) fn load_mlm_head(
    safetensors_path: &Path,
    device: &'static Device,
    config: &BertConfig,
) -> EmbeddingResult<MlmHeadWeights> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, device).map_err(
            |e| EmbeddingError::GpuError {
                message: format!("SparseModel MLM head safetensors load failed: {}", e),
            },
        )?
    };

    // MLM head weights use "cls.predictions" prefix
    let dense_weight = vb
        .get(
            (config.hidden_size, config.hidden_size),
            "cls.predictions.transform.dense.weight",
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM dense weight load failed: {}", e),
        })?;

    let dense_bias = vb
        .get(
            (config.hidden_size,),
            "cls.predictions.transform.dense.bias",
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM dense bias load failed: {}", e),
        })?;

    let layer_norm_weight = vb
        .get(
            (config.hidden_size,),
            "cls.predictions.transform.LayerNorm.weight",
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM LayerNorm weight load failed: {}", e),
        })?;

    let layer_norm_bias = vb
        .get(
            (config.hidden_size,),
            "cls.predictions.transform.LayerNorm.bias",
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM LayerNorm bias load failed: {}", e),
        })?;

    // Decoder weight - note: BERT MLM ties weights with word embeddings
    // Try to load decoder weight, fallback to word embeddings if not present
    let decoder_weight = vb
        .get(
            (config.vocab_size, config.hidden_size),
            "cls.predictions.decoder.weight",
        )
        .or_else(|_| {
            vb.get(
                (config.vocab_size, config.hidden_size),
                "embeddings.word_embeddings.weight",
            )
        })
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM decoder weight load failed: {}", e),
        })?;

    let decoder_bias = vb
        .get((config.vocab_size,), "cls.predictions.bias")
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("SparseModel MLM decoder bias load failed: {}", e),
        })?;

    Ok(MlmHeadWeights {
        dense_weight,
        dense_bias,
        layer_norm_weight,
        layer_norm_bias,
        decoder_weight,
        decoder_bias,
    })
}
