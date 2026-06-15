//! GPU memory management for RTX 5090 32GB VRAM.
//!
//! # Design
//!
//! Memory management strategy for 32GB VRAM:
//!
//! | Pool | Size | Purpose |
//! |------|------|---------|
//! | Model Weights | 16GB | Pretrained model parameters |
//! | Activation Cache | 8GB | Intermediate activations |
//! | Working Memory | 6GB | Batch processing buffers |
//! | Reserved | 2GB | System overhead, fragmentation |
//!
//! # Usage
//!
//! ```rust,no_run
//! use context_graph_embeddings::gpu::{VramTracker, MemoryError};
//!
//! fn allocate_model_weights() -> Result<(), MemoryError> {
//!     // Create tracker for RTX 5090 with 32GB VRAM
//!     let mut tracker = VramTracker::new(32 * 1024 * 1024 * 1024);
//!
//!     // Allocate 16GB for model weights
//!     tracker.allocate("model_weights", 16 * 1024 * 1024 * 1024)?;
//!
//!     // Check available memory
//!     let available_gb = tracker.available() / (1024 * 1024 * 1024);
//!     println!("Available: {} GB", available_gb);
//!
//!     // Deallocate when done
//!     let freed = tracker.deallocate("model_weights");
//!     println!("Freed {} bytes", freed);
//!
//!     Ok(())
//! }
//! ```
//!
//! For thread-safe concurrent access, use [`GpuMemoryPool`]:
//!
//! ```rust,no_run
//! use context_graph_embeddings::gpu::GpuMemoryPool;
//!
//! // Thread-safe pool with RTX 5090 default (32GB)
//! let pool = GpuMemoryPool::rtx_5090();
//!
//! // Safe concurrent allocations
//! if pool.allocate("activation_cache", 8 * 1024 * 1024 * 1024).is_ok() {
//!     println!("Allocated activation cache, {} bytes available", pool.available().unwrap());
//! }
//! ```

mod budget;
mod error;
mod pool;
mod pressure;
mod slots;
mod stats;
mod tracker;

// Re-export all public types
pub use budget::MemoryBudget;
pub use error::MemoryError;
pub use pool::GpuMemoryPool;
pub use stats::MemoryStats;
pub use tracker::VramTracker;

// These are currently unused but kept for future use
#[allow(unused_imports)]
pub use pressure::MemoryPressure;
#[allow(unused_imports)]
pub use slots::{ModelSlot, ModelSlotManager, MODEL_BUDGET_BYTES};
