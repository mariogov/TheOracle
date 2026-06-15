//! Warm Loading Data Types for GPU Weight Management.
//!
//! This module defines the data structures used for loading model weights into GPU memory
//! during warm startup. All types enforce fail-fast validation per Constitution AP-007.
//!
//! # Constitution Alignment
//!
//! - **AP-007**: No Stub Data in Production - All fields contain REAL validated data
//! - **REQ-WARM-003**: Non-evictable VRAM allocation
//! - **REQ-WARM-005**: Weight integrity verification via SHA256 checksums
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**
//!
//! - All constructors panic on invalid data (null pointers, zero checksums, empty collections)
//! - No silent defaults or fallback values
//! - Validation happens at construction time, not runtime
//!
//! # Critical: No Simulation
//!
//! These types are designed to hold REAL data from actual GPU operations:
//! - `gpu_ptr` must be a real cudaMalloc pointer (never 0x0)
//! - `checksum` must be a real SHA256 hash (never all zeros)
//! - `tensors` must contain real GpuTensor instances backed by GPU memory

mod inference;
mod metadata;
mod vram;
mod weights;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::inference::InferenceValidation;
pub use self::metadata::{TensorMetadata, WarmLoadResult};
pub use self::vram::VramAllocationTracking;
pub use self::weights::LoadedModelWeights;

// =============================================================================
// COMPILE-TIME ASSERTIONS
// =============================================================================

/// Compile-time check: Checksum size must be 32 bytes (SHA256)
const _: () = assert!(
    std::mem::size_of::<[u8; 32]>() == 32,
    "Checksum must be exactly 32 bytes for SHA256"
);
