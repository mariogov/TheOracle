//! Device placement options for model inference.
//!
//! Determines where model weights are loaded and inference is executed.

use serde::{Deserialize, Serialize};

/// Device placement options for model inference.
///
/// Determines where model weights are loaded and inference is executed.
///
/// # Serialization
///
/// Serializes as snake_case strings for TOML compatibility:
/// - `"cpu"` -> `DevicePlacement::Cpu`
/// - `"auto"` -> `DevicePlacement::Auto`
/// - `{ "cuda": 0 }` -> `DevicePlacement::Cuda(0)`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DevicePlacement {
    /// CPU-only inference. Slower but always available.
    Cpu,

    /// Specific CUDA device by index.
    /// Index 0 is the primary GPU (RTX 5090).
    Cuda(u32),

    /// Auto-select best available device.
    /// Prefers CUDA if available, falls back to CPU.
    #[default]
    Auto,
}

impl DevicePlacement {
    /// Returns true if this placement requires a GPU.
    pub fn requires_gpu(&self) -> bool {
        matches!(self, DevicePlacement::Cuda(_))
    }

    /// Returns the CUDA device ID if specified, None otherwise.
    pub fn cuda_device_id(&self) -> Option<u32> {
        match self {
            DevicePlacement::Cuda(id) => Some(*id),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_placement_default_is_auto() {
        let placement = DevicePlacement::default();
        assert_eq!(placement, DevicePlacement::Auto);
    }

    #[test]
    fn test_device_placement_requires_gpu() {
        assert!(!DevicePlacement::Cpu.requires_gpu());
        assert!(DevicePlacement::Cuda(0).requires_gpu());
        assert!(!DevicePlacement::Auto.requires_gpu());
    }

    #[test]
    fn test_device_placement_cuda_device_id() {
        assert_eq!(DevicePlacement::Cpu.cuda_device_id(), None);
        assert_eq!(DevicePlacement::Cuda(0).cuda_device_id(), Some(0));
        assert_eq!(DevicePlacement::Cuda(1).cuda_device_id(), Some(1));
        assert_eq!(DevicePlacement::Auto.cuda_device_id(), None);
    }

    #[test]
    fn test_device_placement_serde_roundtrip() {
        let placements = [
            DevicePlacement::Cpu,
            DevicePlacement::Cuda(0),
            DevicePlacement::Cuda(1),
            DevicePlacement::Auto,
        ];

        for placement in placements {
            let json = serde_json::to_string(&placement).unwrap();
            let restored: DevicePlacement = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, placement);
        }
    }

    #[test]
    fn test_device_placement_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<DevicePlacement>();
    }
}
