//! Diagnostic Service
//!
//! Provides methods to generate diagnostic reports in various formats,
//! suitable for both human consumption and automated monitoring.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::warm::error::{WarmError, WarmResult};
use crate::warm::loader::WarmLoader;
use crate::warm::state::WarmModelState;

use super::helpers::format_bytes;
use super::output::dump_report_to_stderr;
use super::types::{
    ErrorDiagnostic, GpuDiagnostics, MemoryDiagnostics, ModelDiagnostic, SystemInfo,
    WarmDiagnosticReport,
};

/// Diagnostic service for warm model loading.
///
/// Provides methods to generate diagnostic reports in various formats,
/// suitable for both human consumption and automated monitoring.
///
/// # Design
///
/// This is a stateless service with only static methods. All diagnostic
/// information is gathered from the [`WarmLoader`] instance.
pub struct WarmDiagnostics;

impl WarmDiagnostics {
    /// Generate a complete diagnostic report from the loader state.
    ///
    /// Captures:
    /// - System information
    /// - GPU information (if available)
    /// - Memory pool status
    /// - Per-model state and allocations
    /// - Any errors from failed models
    #[must_use]
    pub fn generate_report(loader: &WarmLoader) -> WarmDiagnosticReport {
        let mut report = WarmDiagnosticReport::empty();

        // Gather system information
        report.system = SystemInfo::gather();

        // Gather GPU information
        if let Some(gpu_info) = loader.gpu_info() {
            let used: usize = loader
                .memory_pools()
                .list_model_allocations()
                .iter()
                .map(|a| a.size_bytes)
                .sum();
            let available = loader
                .memory_pools()
                .model_pool_capacity()
                .saturating_sub(used);
            report.gpu = Some(GpuDiagnostics::from_gpu_info(gpu_info, available));
        }

        // Gather memory pool information
        report.memory = Self::gather_memory_info(loader);

        // Gather per-model information
        Self::gather_model_info(loader, &mut report);

        report
    }

    /// Gather memory pool diagnostics.
    fn gather_memory_info(loader: &WarmLoader) -> MemoryDiagnostics {
        let pools = loader.memory_pools();
        let allocations = pools.list_model_allocations();
        let model_pool_used: usize = allocations.iter().map(|a| a.size_bytes).sum();

        MemoryDiagnostics {
            model_pool_capacity_bytes: pools.model_pool_capacity(),
            model_pool_used_bytes: model_pool_used,
            working_pool_capacity_bytes: pools.working_pool_capacity(),
            working_pool_used_bytes: pools
                .working_pool_capacity()
                .saturating_sub(pools.available_working_bytes()),
            total_allocations: allocations.len(),
        }
    }

    /// Gather per-model diagnostics.
    fn gather_model_info(loader: &WarmLoader, report: &mut WarmDiagnosticReport) {
        let registry = match loader.registry().read() {
            Ok(r) => r,
            Err(_) => return, // Lock poisoned, return partial report
        };

        for model_id in crate::warm::registry::EMBEDDING_MODEL_IDS.iter() {
            if let Some(entry) = registry.get_entry(model_id) {
                let handle_info = entry
                    .handle
                    .as_ref()
                    .map(|h| (h.vram_address(), h.allocation_bytes(), h.weight_checksum()));

                let diagnostic = ModelDiagnostic::from_state(
                    model_id,
                    &entry.state,
                    entry.expected_bytes,
                    handle_info,
                );

                // Collect errors from failed models
                if let WarmModelState::Failed {
                    error_code,
                    error_message,
                } = &entry.state
                {
                    report.errors.push(ErrorDiagnostic {
                        error_code: format!("ERR-WARM-MODEL-{}", error_code),
                        category: "MODEL".to_string(),
                        message: error_message.clone(),
                        exit_code: *error_code as i32,
                    });
                }

                report.models.push(diagnostic);
            }
        }
    }

    /// Generate a JSON string representation of the diagnostic report.
    pub fn to_json(loader: &WarmLoader) -> WarmResult<String> {
        let report = Self::generate_report(loader);
        serde_json::to_string_pretty(&report).map_err(|e| WarmError::DiagnosticDumpFailed {
            reason: format!("JSON serialization failed: {}", e),
        })
    }

    /// Write the diagnostic report to a file.
    pub fn write_to_file(loader: &WarmLoader, path: &Path) -> WarmResult<()> {
        let json = Self::to_json(loader)?;

        let mut file = File::create(path).map_err(|e| WarmError::DiagnosticDumpFailed {
            reason: format!("Failed to create file {}: {}", path.display(), e),
        })?;

        file.write_all(json.as_bytes())
            .map_err(|e| WarmError::DiagnosticDumpFailed {
                reason: format!("Failed to write to file {}: {}", path.display(), e),
            })?;

        tracing::info!("Diagnostic report written to {}", path.display());
        Ok(())
    }

    /// Dump diagnostic report to stderr.
    ///
    /// Used for fatal errors when the system is about to exit.
    pub fn dump_to_stderr(loader: &WarmLoader) {
        let report = Self::generate_report(loader);
        dump_report_to_stderr(&report);
    }

    /// Generate a minimal status line for quick monitoring.
    ///
    /// Format: `WARM: 13/13 models | 24.0GB/24.0GB VRAM | OK`
    #[must_use]
    pub fn status_line(loader: &WarmLoader) -> String {
        let summary = loader.loading_summary();
        let pools = loader.memory_pools();

        let model_pool_used: usize = pools
            .list_model_allocations()
            .iter()
            .map(|a| a.size_bytes)
            .sum();

        let status = if summary.models_failed > 0 {
            format!("ERRORS: {}", summary.models_failed)
        } else if summary.models_warm == summary.total_models && summary.total_models > 0 {
            "OK".to_string()
        } else {
            format!("LOADING: {}/{}", summary.models_warm, summary.total_models)
        };

        format!(
            "WARM: {}/{} models | {}/{} VRAM | {}",
            summary.models_warm,
            summary.total_models,
            format_bytes(model_pool_used),
            format_bytes(pools.model_pool_capacity()),
            status
        )
    }
}
