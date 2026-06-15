//! Error types for GPU model loading operations.
//!
//! Provides detailed context for debugging loading failures:
//! - Model path that failed
//! - Specific layer/weight that failed
//! - Expected vs actual tensor shapes

use thiserror::Error;

/// Error type for model loading operations.
#[derive(Debug, Error)]
pub enum ModelLoadError {
    /// GPU initialization failed.
    #[error("GPU initialization failed: {message}")]
    GpuInitError { message: String },

    /// Model directory does not exist or is not accessible.
    #[error("Model directory not found: {path}")]
    ModelDirectoryNotFound { path: String },

    /// config.json file missing or unreadable.
    #[error("Config file not found at {path}: {source}")]
    ConfigNotFound {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// config.json parsing failed.
    #[error("Config parse error for {path}: {message}")]
    ConfigParseError { path: String, message: String },

    /// model.safetensors file missing.
    #[error("Safetensors file not found at {path}")]
    SafetensorsNotFound { path: String },

    /// Safetensors file loading failed.
    #[error("Failed to load safetensors from {path}: {message}")]
    SafetensorsLoadError { path: String, message: String },

    /// Specific weight tensor not found in safetensors.
    #[error("Weight not found: {weight_name} in {model_path}")]
    WeightNotFound {
        weight_name: String,
        model_path: String,
    },

    /// Weight tensor has unexpected shape.
    #[error("Shape mismatch for {weight_name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        weight_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Candle tensor operation failed.
    #[error("Tensor operation failed for {operation}: {message}")]
    TensorError { operation: String, message: String },

    /// Unsupported model architecture.
    #[error("Unsupported architecture: {architecture} (supported: BERT, MPNet)")]
    UnsupportedArchitecture { architecture: String },
}

impl From<candle_core::Error> for ModelLoadError {
    fn from(err: candle_core::Error) -> Self {
        ModelLoadError::TensorError {
            operation: "candle".to_string(),
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_error_display() {
        let err = ModelLoadError::WeightNotFound {
            weight_name: "encoder.layer.0.attention.self.query.weight".to_string(),
            model_path: "/models/semantic".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("encoder.layer.0.attention.self.query.weight"));
        assert!(msg.contains("/models/semantic"));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let err = ModelLoadError::ShapeMismatch {
            weight_name: "embeddings.word_embeddings.weight".to_string(),
            expected: vec![30522, 768],
            actual: vec![30522, 1024],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("30522"));
        assert!(msg.contains("768"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn test_gpu_init_error_conversion() {
        let err = candle_core::Error::Msg("CUDA not available".to_string());
        let load_err: ModelLoadError = err.into();
        match load_err {
            ModelLoadError::TensorError { operation, message } => {
                assert_eq!(operation, "candle");
                assert!(message.contains("CUDA"));
            }
            _ => panic!("Expected TensorError"),
        }
    }
}
