//! Batch loading of multiple BERT models from configuration files.
//!
//! Provides functionality to load multiple models from a TOML configuration file.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use super::error::ModelLoadError;
use super::loader::GpuModelLoader;
use super::weights::BertWeights;

/// Load multiple models from a models_config.toml file.
///
/// # Arguments
///
/// * `loader` - The GPU model loader to use
/// * `models_config_path` - Path to models_config.toml
///
/// # Returns
///
/// Map of model name to loaded BertWeights.
pub fn load_models_from_config(
    loader: &GpuModelLoader,
    models_config_path: &Path,
) -> Result<HashMap<String, BertWeights>, ModelLoadError> {
    let content = std::fs::read_to_string(models_config_path).map_err(|e| {
        ModelLoadError::ConfigNotFound {
            path: models_config_path.display().to_string(),
            source: e,
        }
    })?;

    #[derive(Deserialize)]
    struct ModelsConfig {
        models: HashMap<String, ModelEntry>,
    }

    #[derive(Deserialize)]
    struct ModelEntry {
        path: String,
        #[allow(dead_code)]
        repo: String,
    }

    let config: ModelsConfig =
        toml::from_str(&content).map_err(|e| ModelLoadError::ConfigParseError {
            path: models_config_path.display().to_string(),
            message: e.to_string(),
        })?;

    let mut loaded = HashMap::new();

    for (name, entry) in config.models {
        let model_dir = Path::new(&entry.path);

        // Check if model.safetensors exists
        let safetensors_path = model_dir.join("model.safetensors");
        if !safetensors_path.exists() {
            tracing::warn!(
                "Skipping model '{}': no model.safetensors at {}",
                name,
                safetensors_path.display()
            );
            continue;
        }

        match loader.load_bert_weights(model_dir) {
            Ok(weights) => {
                tracing::info!("Loaded model '{}': {} params", name, weights.param_count());
                loaded.insert(name, weights);
            }
            Err(e) => {
                tracing::warn!("Failed to load model '{}': {}", name, e);
            }
        }
    }

    Ok(loaded)
}
