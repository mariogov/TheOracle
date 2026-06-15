//! Basic unit tests for the WarmEmbeddingPipeline.

use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::warm::health::{WarmHealthChecker, WarmHealthStatus};
use crate::warm::integration::WarmEmbeddingPipeline;
use crate::warm::registry::{EMBEDDING_MODEL_IDS, TOTAL_MODEL_COUNT};

use super::{prepare_warmed_pipeline, test_config, test_handle};

#[test]
fn test_pipeline_creation() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Pipeline should exist but not be ready
    assert!(!pipeline.is_ready());
    assert!(!pipeline.is_initialized());
    assert!(pipeline.uptime().is_none());
}

#[test]
fn test_pipeline_warm_all_models() {
    let pipeline = prepare_warmed_pipeline();

    // Now pipeline should be ready
    assert!(pipeline.is_ready());
    assert!(pipeline.is_initialized());
}

#[test]
fn test_pipeline_is_ready() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Initially not ready
    assert!(!pipeline.is_ready());

    // After marking initialized but with no warm models, still not ready
    // because health check will fail
    pipeline.initialized.store(true, Ordering::SeqCst);
    assert!(!pipeline.is_ready()); // Health checker reports not healthy
}

#[test]
fn test_pipeline_health() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    let health = pipeline.health();

    // Initial state should be Loading (models pending)
    assert_eq!(health.status, WarmHealthStatus::Loading);
    assert_eq!(health.models_total, TOTAL_MODEL_COUNT);
    assert_eq!(health.models_warm, 0);
    assert_eq!(health.models_pending, TOTAL_MODEL_COUNT);
    assert_eq!(health.models_failed, 0);
    assert!(health.error_messages.is_empty());
}

#[test]
fn test_pipeline_diagnostics() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    let report = pipeline.diagnostics();

    // Verify report structure
    assert!(!report.timestamp.is_empty());
    assert!(!report.system.hostname.is_empty() || report.system.hostname == "unknown");
    assert_eq!(report.models.len(), TOTAL_MODEL_COUNT);
    assert_eq!(report.warm_count(), 0);
    assert_eq!(report.failed_count(), 0);
}

#[test]
fn test_pipeline_uptime() {
    let config = test_config();
    let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Before initialization, no uptime
    assert!(pipeline.uptime().is_none());

    // Simulate initialization
    pipeline.initialization_time = Some(Instant::now());

    // After initialization, uptime should be tracked
    let uptime = pipeline.uptime();
    assert!(uptime.is_some());
    let _ = uptime.unwrap();
}

#[test]
fn test_pipeline_registry_access() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    let registry = pipeline.registry();

    // Verify we can read the registry
    let guard = registry.read().unwrap();
    assert_eq!(guard.model_count(), TOTAL_MODEL_COUNT);

    // Verify all models are registered
    for model_id in EMBEDDING_MODEL_IDS {
        assert!(guard.get_state(model_id).is_some());
    }
}

#[test]
fn test_create_and_warm() {
    // Note: We can't fully test create_and_warm() because it calls exit()
    // on failure. Instead, we test the happy path components.

    let config = test_config();
    let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Simulate successful loading
    {
        let mut registry = pipeline.registry().write().unwrap();
        for model_id in EMBEDDING_MODEL_IDS.iter() {
            registry.start_loading(model_id).unwrap();
            registry.mark_validating(model_id).unwrap();
            registry.mark_warm(model_id, test_handle()).unwrap();
        }
    }

    // Complete initialization
    pipeline.initialized.store(true, Ordering::SeqCst);
    pipeline.initialization_time = Some(Instant::now());
    pipeline.health_checker = WarmHealthChecker::from_loader(&pipeline.loader);

    // Verify all conditions that create_and_warm would check
    assert!(pipeline.is_ready());
    assert!(pipeline.is_initialized());

    let health = pipeline.health();
    assert_eq!(health.status, WarmHealthStatus::Healthy);
    assert_eq!(health.models_warm, TOTAL_MODEL_COUNT);
}

#[test]
fn test_pipeline_status_line() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    let status = pipeline.status_line();

    // Verify format
    assert!(status.contains("WARM:"));
    assert!(status.contains("models"));
    assert!(status.contains("VRAM"));
    // Initial state should show LOADING
    assert!(status.contains("LOADING:") || status.contains("0/12"));
}

#[test]
fn test_pipeline_loader_access() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    let loader = pipeline.loader();
    let summary = loader.loading_summary();

    assert_eq!(summary.total_models, TOTAL_MODEL_COUNT);
    assert!(!summary.all_warm());
}

#[test]
fn test_pipeline_is_initialized_vs_is_ready() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Initially neither initialized nor ready
    assert!(!pipeline.is_initialized());
    assert!(!pipeline.is_ready());

    // After setting initialized flag, still not ready (health check fails)
    pipeline.initialized.store(true, Ordering::SeqCst);
    assert!(pipeline.is_initialized());
    assert!(!pipeline.is_ready()); // Health check still fails
}
