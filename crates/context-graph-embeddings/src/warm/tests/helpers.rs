//! Helper functions and constants for warm model loading tests.

use crate::warm::config::WarmConfig;
use crate::warm::handle::ModelHandle;

use std::path::PathBuf;

/// One gigabyte in bytes.
pub const GB: usize = 1024 * 1024 * 1024;

/// One megabyte in bytes.
pub const MB: usize = 1024 * 1024;

/// Create a test configuration with valid paths.
#[allow(clippy::field_reassign_with_default)]
pub fn test_config() -> WarmConfig {
    let mut config = WarmConfig::default();
    // Use current directory which exists
    config.model_weights_path = PathBuf::from(".");
    config
}

/// Create a test ModelHandle with specified bytes.
pub fn test_handle(bytes: usize) -> ModelHandle {
    ModelHandle::new(0x1000_0000, bytes, 0, 0xDEAD_BEEF)
}

/// Create a test ModelHandle with custom address and checksum.
pub fn test_handle_full(address: u64, bytes: usize, device: u32, checksum: u64) -> ModelHandle {
    ModelHandle::new(address, bytes, device, checksum)
}
