//! Advanced unit tests for the WarmEmbeddingPipeline.

use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::warm::health::WarmHealthStatus;
use crate::warm::integration::WarmEmbeddingPipeline;

use super::test_config;

/// Test that calling warm() on an already-initialized pipeline is a no-op.
///
/// CUDA feature required because warm() is only defined with cuda feature.
#[test]
#[cfg(feature = "candle")]
fn test_pipeline_warm_already_initialized() {
    let config = test_config();
    let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Simulate initialization
    pipeline.initialized.store(true, Ordering::SeqCst);

    // Second warm() should be a no-op
    let result = pipeline.warm();
    assert!(result.is_ok());
}

#[test]
fn test_pipeline_health_check_unhealthy() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Fail one model
    {
        let mut registry = pipeline.registry().write().unwrap();
        registry.start_loading("E1_Semantic").unwrap();
        registry
            .mark_failed("E1_Semantic", 102, "CUDA allocation failed")
            .unwrap();
    }

    let health = pipeline.health();
    assert_eq!(health.status, WarmHealthStatus::Unhealthy);
    assert_eq!(health.models_failed, 1);
    assert!(!health.error_messages.is_empty());
    assert!(health.error_messages[0].contains("E1_Semantic"));
}

#[test]
fn test_pipeline_uptime_increases() {
    let config = test_config();
    let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Set initialization time
    pipeline.initialization_time = Some(Instant::now());

    let uptime1 = pipeline.uptime().unwrap();

    // Wait a tiny bit
    std::thread::sleep(std::time::Duration::from_millis(1));

    let uptime2 = pipeline.uptime().unwrap();
    assert!(uptime2 > uptime1);
}

#[test]
fn test_pipeline_diagnostics_with_failed_models() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Fail some models
    {
        let mut registry = pipeline.registry().write().unwrap();
        registry.start_loading("E1_Semantic").unwrap();
        registry.mark_failed("E1_Semantic", 102, "Error 1").unwrap();

        registry.start_loading("E2_TemporalRecent").unwrap();
        registry
            .mark_failed("E2_TemporalRecent", 104, "Error 2")
            .unwrap();
    }

    let report = pipeline.diagnostics();
    assert_eq!(report.failed_count(), 2);
    assert!(report.has_errors());
    assert_eq!(report.errors.len(), 2);
}

#[test]
fn test_pipeline_concurrent_health_checks() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    // Simulate concurrent health checks (should not deadlock)
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let health_checker = pipeline.health_checker.clone();
            std::thread::spawn(move || {
                for _ in 0..10 {
                    let _ = health_checker.check();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_pipeline_memory_metrics_in_health() {
    let config = test_config();
    let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

    let health = pipeline.health();

    // VRAM metrics should be populated
    assert!(health.vram_total_bytes() > 0);
    assert!(health.working_memory_available_bytes > 0);

    // Initially no VRAM allocated
    assert_eq!(health.vram_allocated_bytes, 0);
}
