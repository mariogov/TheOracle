//! BatchProcessor: Async multi-model batch orchestration.
//!
//! Manages per-model queues and worker tasks that process embedding requests
//! in optimal batch sizes for GPU efficiency.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                       BatchProcessor                                │
//! │  ┌───────────────────────────────────────────────────────────────┐  │
//! │  │                      Worker Task                               │  │
//! │  │  request_rx ──► Per-Model Queues ──► should_flush() ──►       │  │
//! │  │                      │                                         │  │
//! │  │                      ▼                                         │  │
//! │  │             Semaphore (max_concurrent_batches)                 │  │
//! │  │                      │                                         │  │
//! │  │                      ▼                                         │  │
//! │  │             process_batch(batch, registry)                     │  │
//! │  │                      │                                         │  │
//! │  │                      ▼                                         │  │
//! │  │             batch.complete(results)                            │  │
//! │  └───────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: All errors propagate via EmbeddingError
//! - **FAIL FAST**: Invalid state = immediate error with context
//! - **ASYNC NATIVE**: Uses tokio oneshot/mpsc channels
//! - **THREAD SAFE**: Arc<RwLock<>> for shared state, atomics for stats
//!
//! # Module Structure
//!
//! - `config` - Configuration types and validation
//! - `stats` - Statistics types for metrics tracking
//! - `worker` - Worker loop and batch processing logic
//! - `core` - Main BatchProcessor struct and lifecycle methods
//! - `submit` - Request submission API

mod config;
mod core;
mod stats;
mod submit;
mod worker;

// Re-export public types for backwards compatibility
pub use config::BatchProcessorConfig;
pub use core::BatchProcessor;
pub use stats::BatchProcessorStats;
