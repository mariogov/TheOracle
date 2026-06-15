//! Tensor loading utilities for BERT model weights.
//!
//! Provides helper functions for loading and validating tensors from VarBuilder.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::error::ModelLoadError;

/// Get a tensor from VarBuilder with shape validation.
pub fn get_tensor(
    vb: &VarBuilder,
    name: &str,
    expected_shape: &[usize],
    model_path: &str,
) -> Result<Tensor, ModelLoadError> {
    let tensor = vb.get(expected_shape, name).map_err(|e| {
        // Check if it's a shape error or missing tensor
        let err_str = e.to_string();
        if err_str.contains("shape") || err_str.contains("Shape") {
            ModelLoadError::ShapeMismatch {
                weight_name: name.to_string(),
                expected: expected_shape.to_vec(),
                actual: vec![], // We don't have access to actual shape in error
            }
        } else {
            ModelLoadError::WeightNotFound {
                weight_name: name.to_string(),
                model_path: model_path.to_string(),
            }
        }
    })?;

    // Verify shape matches exactly
    let actual_shape: Vec<usize> = tensor.dims().to_vec();
    if actual_shape != expected_shape {
        return Err(ModelLoadError::ShapeMismatch {
            weight_name: name.to_string(),
            expected: expected_shape.to_vec(),
            actual: actual_shape,
        });
    }

    Ok(tensor)
}

/// Try to get a tensor, returning a zero tensor if not found.
///
/// This is used for optional weights like token_type_embeddings which may not
/// exist in all model architectures (e.g., MPNet doesn't have token type embeddings).
///
/// # Arguments
/// * `vb` - VarBuilder to load from
/// * `name` - Weight tensor name
/// * `expected_shape` - Expected tensor shape
/// * `device` - Device to create zero tensor on if weight not found
///
/// # Returns
/// The tensor if found, or a zero tensor of the expected shape if not found.
pub fn get_tensor_or_zeros(
    vb: &VarBuilder,
    name: &str,
    expected_shape: &[usize],
    device: &Device,
) -> Result<Tensor, ModelLoadError> {
    match vb.get(expected_shape, name) {
        Ok(tensor) => {
            // Verify shape matches exactly
            let actual_shape: Vec<usize> = tensor.dims().to_vec();
            if actual_shape != expected_shape {
                return Err(ModelLoadError::ShapeMismatch {
                    weight_name: name.to_string(),
                    expected: expected_shape.to_vec(),
                    actual: actual_shape,
                });
            }
            Ok(tensor)
        }
        Err(_) => {
            // Weight not found - create zeros tensor
            // This is expected for MPNet which doesn't use token_type_embeddings
            tracing::debug!(
                "Optional weight '{}' not found, using zeros tensor of shape {:?}",
                name,
                expected_shape
            );
            Tensor::zeros(expected_shape, DType::F32, device).map_err(|e| {
                ModelLoadError::TensorError {
                    operation: format!("create zeros for {}", name),
                    message: e.to_string(),
                }
            })
        }
    }
}
