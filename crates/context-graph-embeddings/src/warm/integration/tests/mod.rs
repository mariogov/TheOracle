//! Unit tests for the WarmEmbeddingPipeline.

mod advanced;
mod basic;

use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::warm::config::WarmConfig;
use crate::warm::handle::ModelHandle;
use crate::warm::health::WarmHealthChecker;
use crate::warm::integration::WarmEmbeddingPipeline;
use crate::warm::registry::EMBEDDING_MODEL_IDS;

/// Create a test config that doesn't require real files.
#[allow(clippy::field_reassign_with_default)]
pub(crate) fn test_config() -> WarmConfig {
    let mut config = WarmConfig::default();
    config.enable_test_inference = true;
    config
}

/// Helper to create a test handle
pub(crate) fn test_handle() -> ModelHandle {
    ModelHandle::new(0x1000_0000, 512 * 1024 * 1024, 0, 0xDEAD_BEEF)
}

/// Helper to prepare a pipeline with all models warmed
pub(crate) fn prepare_warmed_pipeline() -> WarmEmbeddingPipeline {
    let config = test_config();
    let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Manually transition all models to Warm state (simulating load)
    {
        let mut registry = pipeline.registry().write().unwrap();
        for model_id in EMBEDDING_MODEL_IDS.iter() {
            registry.start_loading(model_id).unwrap();
            registry.mark_validating(model_id).unwrap();
            registry.mark_warm(model_id, test_handle()).unwrap();
        }
    }

    // Manually set initialized flag (since we bypassed normal loading)
    pipeline.initialized.store(true, Ordering::SeqCst);
    pipeline.initialization_time = Some(Instant::now());

    // Recreate health checker to pick up new state
    pipeline.health_checker = WarmHealthChecker::from_loader(&pipeline.loader);

    pipeline
}
