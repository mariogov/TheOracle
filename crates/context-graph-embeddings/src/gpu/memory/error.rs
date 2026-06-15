//! Memory allocation errors for GPU operations.

/// Memory allocation errors.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Insufficient VRAM for requested allocation.
    #[error("Out of GPU memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Number of bytes requested.
        requested: usize,
        /// Number of bytes available.
        available: usize,
    },

    /// Lock poisoned (thread panic while holding lock).
    #[error("Memory pool lock poisoned")]
    LockPoisoned,
}
