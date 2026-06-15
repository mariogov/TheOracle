//! End-to-End tests for Context Graph hook system
//!
//! # CRITICAL: E2E vs Integration Tests
//! - **Integration tests**: Directly invoke CLI binary via std::process::Command
//! - **E2E tests**: Execute ACTUAL SHELL SCRIPTS that then invoke CLI binary
//!
//! E2E tests verify the complete Claude Code -> Shell Script -> CLI -> MCP -> Database flow.
//!
//! # NO MOCKS - Real Components Only
//! - Real shell scripts (.claude/hooks/*.sh)
//! - Real CLI binary (target/release/context-graph-cli)
//! - Real RocksDB database (temp directories)
//!
//! # Test Categories
//! - `full_session_test`: Complete session workflow via shell scripts
//! - `error_recovery_test`: Error handling and exit codes from shell scripts
//!
//! # Running Tests
//! ```bash
//! # Build CLI first
//! cargo build --release -p context-graph-cli
//!
//! # Run all E2E tests (single-threaded for isolation)
//! cargo test --package context-graph-cli --test e2e -- --test-threads=1 --nocapture
//!
//! # Run specific test
//! cargo test --package context-graph-cli --test e2e test_e2e_full_session_workflow -- --nocapture
//! ```
//!
//! # Constitution References
//! - REQ-HOOKS-45: E2E tests with real MCP
//! - REQ-HOOKS-46: E2E tests simulate Claude Code
//! - REQ-HOOKS-47: No mock data in any tests
//! - AP-26: Exit codes (0-6)
//! - AP-50: Native hooks only

pub mod error_recovery_test;
pub mod full_session_test;
pub mod helpers;
