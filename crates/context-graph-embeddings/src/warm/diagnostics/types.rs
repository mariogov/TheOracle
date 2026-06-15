//! Diagnostic Report Structures
//!
//! Data types for representing diagnostic information about the warm model
//! loading system. All types are serializable for JSON output.

use serde::{Deserialize, Serialize};

use crate::warm::cuda_alloc::GpuInfo;
use crate::warm::error::WarmError;
use crate::warm::state::WarmModelState;

use super::helpers::current_timestamp;

/// Complete diagnostic report for the warm model loading system.
///
/// Contains all information needed to diagnose loading issues, including
/// system info, GPU status, memory usage, and per-model state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmDiagnosticReport {
    /// ISO 8601 timestamp when report was generated.
    pub timestamp: String,
    /// System information.
    pub system: SystemInfo,
    /// GPU information (None if no GPU available).
    pub gpu: Option<GpuDiagnostics>,
    /// Memory pool status.
    pub memory: MemoryDiagnostics,
    /// Per-model diagnostic information.
    pub models: Vec<ModelDiagnostic>,
    /// Any errors encountered during loading.
    pub errors: Vec<ErrorDiagnostic>,
}

impl WarmDiagnosticReport {
    /// Create an empty diagnostic report with current timestamp.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            timestamp: current_timestamp(),
            system: SystemInfo::default(),
            gpu: None,
            memory: MemoryDiagnostics::default(),
            models: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Check if any errors were recorded.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get the count of warm models.
    #[must_use]
    pub fn warm_count(&self) -> usize {
        self.models.iter().filter(|m| m.state == "Warm").count()
    }

    /// Get the count of failed models.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.models.iter().filter(|m| m.state == "Failed").count()
    }
}

/// System information for diagnostic context.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemInfo {
    /// System hostname.
    pub hostname: String,
    /// Operating system description.
    pub os: String,
    /// System uptime in seconds.
    pub uptime_seconds: f64,
}

impl SystemInfo {
    /// Gather system information from the current environment.
    #[must_use]
    pub fn gather() -> Self {
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| {
                std::fs::read_to_string("/etc/hostname")
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| "unknown".to_string())
            });

        let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
        let uptime_seconds = 0.0;

        Self {
            hostname,
            os,
            uptime_seconds,
        }
    }
}

/// GPU diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDiagnostics {
    /// CUDA device ID.
    pub device_id: u32,
    /// GPU model name.
    pub name: String,
    /// Compute capability as string (e.g., "12.0").
    pub compute_capability: String,
    /// Total VRAM in bytes.
    pub total_vram_bytes: usize,
    /// Available (free) VRAM in bytes.
    pub available_vram_bytes: usize,
    /// CUDA driver version.
    pub driver_version: String,
}

impl GpuDiagnostics {
    /// Create GPU diagnostics from a [`GpuInfo`] structure.
    #[must_use]
    pub fn from_gpu_info(info: &GpuInfo, available_bytes: usize) -> Self {
        Self {
            device_id: info.device_id,
            name: info.name.clone(),
            compute_capability: info.compute_capability_string(),
            total_vram_bytes: info.total_memory_bytes,
            available_vram_bytes: available_bytes,
            driver_version: info.driver_version.clone(),
        }
    }
}

/// Memory pool diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryDiagnostics {
    /// Model pool capacity in bytes.
    pub model_pool_capacity_bytes: usize,
    /// Model pool used bytes.
    pub model_pool_used_bytes: usize,
    /// Working pool capacity in bytes.
    pub working_pool_capacity_bytes: usize,
    /// Working pool used bytes.
    pub working_pool_used_bytes: usize,
    /// Total number of model allocations.
    pub total_allocations: usize,
}

/// Per-model diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiagnostic {
    /// Model identifier (e.g., "E1_Semantic").
    pub model_id: String,
    /// Current state as string (e.g., "Warm", "Loading", "Failed").
    pub state: String,
    /// Expected size in bytes.
    pub expected_bytes: usize,
    /// Allocated bytes (if loaded).
    pub allocated_bytes: Option<usize>,
    /// VRAM pointer as hex string (if loaded).
    pub vram_ptr: Option<String>,
    /// Weight checksum as hex string (if loaded).
    pub checksum: Option<String>,
    /// Error message (if failed).
    pub error_message: Option<String>,
}

impl ModelDiagnostic {
    /// Create a diagnostic entry for a model from registry state.
    #[must_use]
    pub fn from_state(
        model_id: &str,
        state: &WarmModelState,
        expected_bytes: usize,
        handle_info: Option<(u64, usize, u64)>,
    ) -> Self {
        let state_str = match state {
            WarmModelState::Pending => "Pending".to_string(),
            WarmModelState::Loading {
                progress_percent, ..
            } => {
                format!("Loading ({}%)", progress_percent)
            }
            WarmModelState::Validating => "Validating".to_string(),
            WarmModelState::Warm => "Warm".to_string(),
            WarmModelState::Failed { .. } => "Failed".to_string(),
        };

        let error_message = match state {
            WarmModelState::Failed { error_message, .. } => Some(error_message.clone()),
            _ => None,
        };

        let (allocated_bytes, vram_ptr, checksum) = match handle_info {
            Some((ptr, size, chk)) => (
                Some(size),
                Some(format!("0x{:016x}", ptr)),
                Some(format!("0x{:016X}", chk)),
            ),
            None => (None, None, None),
        };

        Self {
            model_id: model_id.to_string(),
            state: state_str,
            expected_bytes,
            allocated_bytes,
            vram_ptr,
            checksum,
            error_message,
        }
    }
}

/// Error diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDiagnostic {
    /// Structured error code (e.g., "ERR-WARM-CUDA-INIT").
    pub error_code: String,
    /// Error category (e.g., "CUDA", "MODEL", "VRAM").
    pub category: String,
    /// Human-readable error message.
    pub message: String,
    /// Process exit code for this error.
    pub exit_code: i32,
}

impl ErrorDiagnostic {
    /// Create an error diagnostic from a [`WarmError`].
    #[must_use]
    pub fn from_error(error: &WarmError) -> Self {
        Self {
            error_code: error.error_code().to_string(),
            category: error.category().to_string(),
            message: error.to_string(),
            exit_code: error.exit_code(),
        }
    }
}
