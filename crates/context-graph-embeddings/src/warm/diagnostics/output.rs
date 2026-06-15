//! Diagnostic Output Formatters
//!
//! Functions for outputting diagnostic reports in various formats
//! including stderr dump for fatal errors.

use super::helpers::format_bytes;
use super::types::WarmDiagnosticReport;

/// Dump diagnostic report to stderr.
///
/// Used for fatal errors when the system is about to exit.
/// This is a best-effort operation that should never fail.
pub fn dump_report_to_stderr(report: &WarmDiagnosticReport) {
    eprintln!("\n=== WARM MODEL LOADING DIAGNOSTIC DUMP ===");
    eprintln!("Timestamp: {}", report.timestamp);
    eprintln!();

    // System info
    eprintln!("SYSTEM:");
    eprintln!("  Hostname: {}", report.system.hostname);
    eprintln!("  OS: {}", report.system.os);
    eprintln!();

    // GPU info
    if let Some(gpu) = &report.gpu {
        eprintln!("GPU:");
        eprintln!("  Device: {} (ID: {})", gpu.name, gpu.device_id);
        eprintln!("  Compute Capability: {}", gpu.compute_capability);
        eprintln!("  Total VRAM: {}", format_bytes(gpu.total_vram_bytes));
        eprintln!(
            "  Available VRAM: {}",
            format_bytes(gpu.available_vram_bytes)
        );
        eprintln!("  Driver Version: {}", gpu.driver_version);
    } else {
        eprintln!("GPU: Not available");
    }
    eprintln!();

    // Memory info
    dump_memory_info(report);

    // Model status
    dump_model_status(report);

    // Errors
    dump_errors(report);

    eprintln!("=== END DIAGNOSTIC DUMP ===\n");
}

/// Dump memory pool information to stderr.
fn dump_memory_info(report: &WarmDiagnosticReport) {
    eprintln!("MEMORY:");
    eprintln!(
        "  Model Pool: {} / {} ({:.1}%)",
        format_bytes(report.memory.model_pool_used_bytes),
        format_bytes(report.memory.model_pool_capacity_bytes),
        calculate_percentage(
            report.memory.model_pool_used_bytes,
            report.memory.model_pool_capacity_bytes
        )
    );
    eprintln!(
        "  Working Pool: {} / {} ({:.1}%)",
        format_bytes(report.memory.working_pool_used_bytes),
        format_bytes(report.memory.working_pool_capacity_bytes),
        calculate_percentage(
            report.memory.working_pool_used_bytes,
            report.memory.working_pool_capacity_bytes
        )
    );
    eprintln!("  Total Allocations: {}", report.memory.total_allocations);
    eprintln!();
}

/// Dump per-model status to stderr.
fn dump_model_status(report: &WarmDiagnosticReport) {
    eprintln!("MODELS ({} total):", report.models.len());
    for model in &report.models {
        let status_icon = match model.state.as_str() {
            "Warm" => "[OK]",
            "Failed" => "[FAIL]",
            s if s.starts_with("Loading") => "[...]",
            "Validating" => "[VAL]",
            _ => "[---]",
        };

        eprintln!(
            "  {} {} - {} (expected: {})",
            status_icon,
            model.model_id,
            model.state,
            format_bytes(model.expected_bytes)
        );

        if let Some(ptr) = &model.vram_ptr {
            let allocated_str = model
                .allocated_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "N/A".to_string());
            eprintln!("      VRAM: {} ({} allocated)", ptr, allocated_str);
        }

        if let Some(err) = &model.error_message {
            eprintln!("      ERROR: {}", err);
        }
    }
    eprintln!();
}

/// Dump error information to stderr.
fn dump_errors(report: &WarmDiagnosticReport) {
    if !report.errors.is_empty() {
        eprintln!("ERRORS ({}):", report.errors.len());
        for error in &report.errors {
            eprintln!(
                "  [{}] {} (exit code {})",
                error.category, error.error_code, error.exit_code
            );
            eprintln!("      {}", error.message);
        }
        eprintln!();
    }
}

/// Calculate percentage with zero-division protection.
#[inline]
fn calculate_percentage(used: usize, capacity: usize) -> f64 {
    if capacity > 0 {
        (used as f64 / capacity as f64) * 100.0
    } else {
        0.0
    }
}
