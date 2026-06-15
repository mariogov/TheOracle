//! Integration tests for Context Graph CLI hook lifecycle
//!
//! # Test Architecture
//! Tests execute the REAL CLI binary against REAL RocksDB databases.
//! NO MOCKS - all operations are verified at the physical storage level.
//!
//! # Test Categories
//! - `hook_lifecycle_test`: Full session lifecycle (start -> tools -> end)
//! - `exit_code_test`: Error conditions and exit code verification
//! - `timeout_test`: Timing budget compliance
//!
//! # Running Tests
//! ```bash
//! # Build CLI first
//! cargo build --release -p context-graph-cli
//!
//! # Run all integration tests
//! cargo test --package context-graph-cli --test integration -- --test-threads=1 --nocapture
//!
//! # Run specific suite
//! cargo test --package context-graph-cli --test integration hook_lifecycle -- --nocapture
//! ```
//!
//! # Constitution References
//! - REQ-HOOKS-43: Integration tests for lifecycle
//! - REQ-HOOKS-47: No mock data in any tests

pub mod exit_code_test;
pub mod helpers;
pub mod hook_lifecycle_test;
pub mod timeout_test;
