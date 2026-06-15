//! Pipeline Integration Module for Warm Model Loading
//!
//! Connects [`WarmLoader`] with the embedding pipeline to provide a unified
//! warmed embedding system with all models pre-loaded in VRAM at startup.
//!
//! This integration implements a **fail-fast** strategy. If any component
//! fails during initialization, the pipeline terminates immediately with
//! an appropriate exit code (101-110).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::warm::config::WarmConfig;
use crate::warm::diagnostics::{WarmDiagnosticReport, WarmDiagnostics};
use crate::warm::error::{WarmError, WarmResult};
use crate::warm::health::{WarmHealthCheck, WarmHealthChecker};
use crate::warm::loader::WarmLoader;
use crate::warm::registry::SharedWarmRegistry;

/// Warmed embedding pipeline with all models pre-loaded in VRAM.
///
/// This is the main entry point for production use of the warm loading system.
/// It integrates [`WarmLoader`], [`WarmHealthChecker`], and [`WarmDiagnostics`]
/// into a unified pipeline that:
///
/// - Loads all 12 embedding models into VRAM at startup
/// - Validates each model with test inference
/// - Provides real-time health monitoring
/// - Generates diagnostic reports on demand
///
/// The pipeline is thread-safe for read operations. Health checks, diagnostics,
/// and registry access can be performed concurrently from multiple threads.
pub struct WarmEmbeddingPipeline {
    /// Main orchestrator for warm model loading.
    pub(crate) loader: WarmLoader,
    /// Health check service for status monitoring.
    pub(crate) health_checker: WarmHealthChecker,
    /// Whether the pipeline has been successfully initialized.
    pub(crate) initialized: AtomicBool,
    /// Timestamp when initialization completed successfully.
    pub(crate) initialization_time: Option<Instant>,
}

impl WarmEmbeddingPipeline {
    /// Create and warm the embedding pipeline.
    ///
    /// This is the main entry point for production use. It creates the loader,
    /// loads all models, validates them, and sets up health monitoring.
    ///
    /// On ANY initialization error, this method logs diagnostics and calls
    /// `std::process::exit()` with the appropriate exit code.
    /// It only returns `Ok(Self)` on complete success.
    ///
    /// # CUDA Requirement
    ///
    /// This function requires the `cuda` feature. CUDA is mandatory per Constitution AP-007.
    #[cfg(feature = "candle")]
    pub fn create_and_warm(config: WarmConfig) -> WarmResult<Self> {
        tracing::info!("Starting WarmEmbeddingPipeline initialization");
        let start_time = Instant::now();

        // Step 1: Create loader
        tracing::info!("Step 1/4: Creating WarmLoader");
        let mut loader = match WarmLoader::new(config) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(
                    exit_code = e.exit_code(),
                    category = %e.category(),
                    "Failed to create WarmLoader"
                );
                Self::handle_fatal_error_static(&e);
            }
        };

        // Step 2: Load all models (FAIL FAST on any error)
        tracing::info!("Step 2/4: Loading all models into VRAM");
        if let Err(e) = loader.load_all_models() {
            tracing::error!(
                exit_code = e.exit_code(),
                category = %e.category(),
                "Failed to load models"
            );
            WarmDiagnostics::dump_to_stderr(&loader);
            Self::handle_fatal_error_static(&e);
        }

        // Step 3: Create health checker
        tracing::info!("Step 3/4: Creating health checker");
        let health_checker = WarmHealthChecker::from_loader(&loader);

        // Step 4: Verify health
        tracing::info!("Step 4/4: Verifying pipeline health");
        if !health_checker.is_healthy() {
            let health = health_checker.check();
            let error = WarmError::ModelValidationFailed {
                model_id: "pipeline".to_string(),
                reason: format!(
                    "Pipeline unhealthy after loading: {} warm, {} failed, {} loading",
                    health.models_warm, health.models_failed, health.models_loading
                ),
                expected_output: Some(format!("Healthy ({} models warm)", health.models_total)),
                actual_output: Some(format!("{:?}", health.status)),
            };

            tracing::error!(
                status = ?health.status,
                models_warm = health.models_warm,
                models_failed = health.models_failed,
                "Pipeline health check failed"
            );

            for msg in &health.error_messages {
                tracing::error!(error = %msg, "Model failure");
            }

            WarmDiagnostics::dump_to_stderr(&loader);
            Self::handle_fatal_error_static(&error);
        }

        let duration = start_time.elapsed();
        tracing::info!(
            duration_ms = duration.as_millis() as u64,
            "WarmEmbeddingPipeline initialization completed successfully"
        );

        Ok(Self {
            loader,
            health_checker,
            initialized: AtomicBool::new(true),
            initialization_time: Some(Instant::now()),
        })
    }

    /// Create a new pipeline without warming (for testing).
    ///
    /// Unlike [`create_and_warm()`](Self::create_and_warm), this method does
    /// NOT load models or exit on error. Use this when you need to test
    /// pipeline behavior without the full loading process.
    pub fn new(config: WarmConfig) -> WarmResult<Self> {
        tracing::debug!("Creating WarmEmbeddingPipeline without warming");

        let loader = WarmLoader::new(config)?;
        let health_checker = WarmHealthChecker::from_loader(&loader);

        Ok(Self {
            loader,
            health_checker,
            initialized: AtomicBool::new(false),
            initialization_time: None,
        })
    }

    /// Warm all models (call after [`new()`](Self::new) if used).
    ///
    /// Loads all 13 models into VRAM and validates them. Unlike
    /// [`create_and_warm()`](Self::create_and_warm), this method returns
    /// an error instead of exiting on failure.
    ///
    /// # CUDA Requirement
    ///
    /// This function requires the `cuda` feature. CUDA is mandatory per Constitution AP-007.
    #[cfg(feature = "candle")]
    pub fn warm(&mut self) -> WarmResult<()> {
        if self.initialized.load(Ordering::SeqCst) {
            tracing::warn!("Pipeline already initialized, skipping warm()");
            return Ok(());
        }

        tracing::info!("Warming pipeline models");

        // Load all models
        self.loader.load_all_models()?;

        // Update health checker with fresh data
        self.health_checker = WarmHealthChecker::from_loader(&self.loader);

        // Verify health
        if !self.health_checker.is_healthy() {
            let health = self.health_checker.check();
            return Err(WarmError::ModelValidationFailed {
                model_id: "pipeline".to_string(),
                reason: format!(
                    "Pipeline unhealthy: {} warm, {} failed",
                    health.models_warm, health.models_failed
                ),
                expected_output: Some("Healthy".to_string()),
                actual_output: Some(format!("{:?}", health.status)),
            });
        }

        self.initialized.store(true, Ordering::SeqCst);
        self.initialization_time = Some(Instant::now());

        tracing::info!("Pipeline warming completed successfully");
        Ok(())
    }

    /// Check if the pipeline is ready for inference.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::SeqCst) && self.health_checker.is_healthy()
    }

    /// Get the current health status.
    #[must_use]
    pub fn health(&self) -> WarmHealthCheck {
        self.health_checker.check()
    }

    /// Get a diagnostic report for the pipeline.
    #[must_use]
    pub fn diagnostics(&self) -> WarmDiagnosticReport {
        WarmDiagnostics::generate_report(&self.loader)
    }

    /// Get a reference to the underlying loader.
    #[must_use]
    pub fn loader(&self) -> &WarmLoader {
        &self.loader
    }

    /// Get a reference to the shared registry.
    #[must_use]
    pub fn registry(&self) -> &SharedWarmRegistry {
        self.loader.registry()
    }

    /// Get the uptime since successful initialization.
    #[must_use]
    pub fn uptime(&self) -> Option<Duration> {
        self.initialization_time.map(|t| t.elapsed())
    }

    /// Get the initialization status.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Get a status line for quick monitoring (e.g., `WARM: 13/13 models | 24.0GB VRAM | OK`).
    #[must_use]
    pub fn status_line(&self) -> String {
        WarmDiagnostics::status_line(&self.loader)
    }

    /// Handle a fatal error by logging diagnostics and exiting.
    fn handle_fatal_error_static(error: &WarmError) -> ! {
        tracing::error!(
            exit_code = error.exit_code(),
            category = %error.category(),
            error_code = %error.error_code(),
            "FATAL: WarmEmbeddingPipeline initialization failed"
        );
        tracing::error!("Error details: {}", error);

        std::process::exit(error.exit_code())
    }
}
