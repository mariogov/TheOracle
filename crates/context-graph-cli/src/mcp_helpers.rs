//! Shared helpers for MCP client operations.
//!
//! This module provides common utilities used by memory commands that interact
//! with the MCP server, eliminating duplication between capture.rs and inject.rs.
//!
//! # Constitution Compliance
//!
//! - AP-11: Check existing utils before creating helpers (this IS the shared utils)
//! - AP-12: No magic numbers - use named constants
//! - AP-14: No .unwrap() in library code
//! - AP-26: Exit code 0=success, 1=error, 2=corruption

use tracing::{debug, error};

use crate::error::CliExitCode;
use crate::mcp_client::{McpClient, McpClientError};

// =============================================================================
// Constants (AP-12: No magic numbers)
// =============================================================================

/// Default session ID when none provided.
pub const DEFAULT_SESSION_ID: &str = "default";

// =============================================================================
// Session ID Resolution
// =============================================================================

/// Resolve session ID from argument or CLAUDE_SESSION_ID environment variable.
///
/// Priority: CLI argument > environment variable > default
///
/// # Arguments
///
/// * `session_id` - Optional session ID from CLI argument
///
/// # Returns
///
/// Resolved session ID string, falling back to DEFAULT_SESSION_ID if none available.
pub fn resolve_session_id(session_id: Option<String>) -> String {
    session_id
        .or_else(|| std::env::var("CLAUDE_SESSION_ID").ok())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            debug!("No session ID available, using '{}'", DEFAULT_SESSION_ID);
            DEFAULT_SESSION_ID.to_string()
        })
}

// =============================================================================
// MCP Error Handling
// =============================================================================

/// Map MCP client error to exit code per AP-26.
///
/// Exit codes:
/// - 1 (Warning): Recoverable errors (server not running, timeout, general errors)
/// - 2 (Blocking): Corruption detected (stale index, data corruption)
///
/// # Arguments
///
/// * `e` - Reference to McpClientError
///
/// # Returns
///
/// Exit code as i32 (1 for warning, 2 for blocking/corruption).
pub fn mcp_error_to_exit_code(e: &McpClientError) -> i32 {
    match e {
        McpClientError::ServerNotRunning { .. } => {
            error!(error = %e, "MCP server not running");
            CliExitCode::Warning as i32
        }
        McpClientError::ConnectionTimeout { .. } => {
            error!(error = %e, "MCP connection timeout");
            CliExitCode::Warning as i32
        }
        McpClientError::RequestTimeout { .. } => {
            error!(error = %e, "MCP request timeout");
            CliExitCode::Warning as i32
        }
        McpClientError::McpError { code, message } => {
            error!(code, message, "MCP tool error");
            if is_corruption_message(message) {
                CliExitCode::Blocking as i32
            } else {
                CliExitCode::Warning as i32
            }
        }
        McpClientError::IoError(e) => {
            error!(error = %e, "MCP IO error");
            CliExitCode::Warning as i32
        }
        McpClientError::JsonError(e) => {
            error!(error = %e, "MCP JSON error");
            CliExitCode::Warning as i32
        }
        McpClientError::NoResult => {
            error!("MCP returned no result");
            CliExitCode::Warning as i32
        }
    }
}

/// Check if an error message indicates storage corruption.
///
/// Used to determine if an MCP error should be treated as blocking (exit 2).
/// Only matches RocksDB-specific corruption patterns — NOT common error words
/// like "not found" (normal for missing memories) or "invalid" (normal for bad input).
///
/// TST-H3 FIX: Delegates to shared `CORRUPTION_INDICATORS` in error.rs
/// instead of maintaining a separate, divergent indicator list.
fn is_corruption_message(message: &str) -> bool {
    crate::error::is_corruption_indicator(message)
}

// =============================================================================
// MCP Server Connection Helpers
// =============================================================================

/// Result of checking MCP server availability.
pub enum ServerCheckResult {
    /// Server is running and ready.
    Running,
    /// Server is not running, includes error message for user.
    NotRunning(String),
    /// Error checking server status.
    CheckFailed(String),
}

/// Check if MCP server is running and return appropriate result.
///
/// This consolidates the repeated server check pattern used in memory commands.
///
/// # Arguments
///
/// * `client` - Reference to McpClient
///
/// # Returns
///
/// ServerCheckResult indicating server status.
pub async fn check_mcp_server(client: &McpClient) -> ServerCheckResult {
    match client.is_server_running().await {
        Ok(true) => {
            debug!("MCP server is running at {}", client.server_address());
            ServerCheckResult::Running
        }
        Ok(false) => {
            let msg = format!(
                "MCP server not running on {}. Start the MCP server first.",
                client.server_address()
            );
            error!(address = %client.server_address(), "MCP server not running");
            ServerCheckResult::NotRunning(msg)
        }
        Err(e) => {
            let msg = format!("Failed to check MCP server: {}", e);
            error!(error = %e, "Failed to check MCP server status");
            ServerCheckResult::CheckFailed(msg)
        }
    }
}

/// Check MCP server and return exit code on failure.
///
/// Convenience function that prints error to stderr and returns exit code
/// if the server is not available.
///
/// # Arguments
///
/// * `client` - Reference to McpClient
///
/// # Returns
///
/// `Ok(())` if server is running, `Err(exit_code)` if not.
pub async fn require_mcp_server(client: &McpClient) -> Result<(), i32> {
    match check_mcp_server(client).await {
        ServerCheckResult::Running => Ok(()),
        ServerCheckResult::NotRunning(msg) => {
            eprintln!("ERROR: {}", msg);
            Err(CliExitCode::Warning as i32)
        }
        ServerCheckResult::CheckFailed(msg) => {
            eprintln!("ERROR: {}", msg);
            Err(CliExitCode::Warning as i32)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_utils::GLOBAL_IDENTITY_LOCK;

    // =========================================================================
    // Session ID Resolution Tests
    // =========================================================================

    #[test]
    fn test_resolve_session_id_from_arg() {
        let result = resolve_session_id(Some("my-session".to_string()));
        assert_eq!(result, "my-session");
    }

    #[test]
    fn test_resolve_session_id_from_env() {
        let _lock = GLOBAL_IDENTITY_LOCK.lock();
        std::env::set_var("CLAUDE_SESSION_ID", "env-session-123");

        let result = resolve_session_id(None);
        assert_eq!(result, "env-session-123");

        std::env::remove_var("CLAUDE_SESSION_ID");
    }

    #[test]
    fn test_resolve_session_id_default() {
        let _lock = GLOBAL_IDENTITY_LOCK.lock();
        std::env::remove_var("CLAUDE_SESSION_ID");

        let result = resolve_session_id(None);
        assert_eq!(result, DEFAULT_SESSION_ID);
    }

    #[test]
    fn test_resolve_session_id_whitespace_uses_default() {
        let _lock = GLOBAL_IDENTITY_LOCK.lock();
        std::env::set_var("CLAUDE_SESSION_ID", "   ");

        let result = resolve_session_id(None);
        assert_eq!(result, DEFAULT_SESSION_ID);

        std::env::remove_var("CLAUDE_SESSION_ID");
    }

    #[test]
    fn test_resolve_session_id_arg_overrides_env() {
        let _lock = GLOBAL_IDENTITY_LOCK.lock();
        std::env::set_var("CLAUDE_SESSION_ID", "env-session");

        let result = resolve_session_id(Some("arg-session".to_string()));
        assert_eq!(result, "arg-session");

        std::env::remove_var("CLAUDE_SESSION_ID");
    }

    // =========================================================================
    // Corruption Detection Tests
    // =========================================================================

    #[test]
    fn test_is_corruption_message() {
        // Real corruption patterns
        assert!(is_corruption_message("data corruption detected"));
        assert!(is_corruption_message("stale index found"));
        assert!(is_corruption_message("checksum mismatch in block 42"));
        assert!(is_corruption_message("bad magic number"));
        assert!(is_corruption_message("CRC Error in SST file"));
        assert!(is_corruption_message("truncated block at offset 1024"));

        // H4 FIX: "not found" is NOT corruption — it's normal for missing memories
        assert!(!is_corruption_message("memory not found"));
        assert!(!is_corruption_message("session not found"));

        // M9 FIX: "invalid" and "malformed" are NOT corruption — they're normal errors
        assert!(!is_corruption_message("invalid request"));
        assert!(!is_corruption_message("invalid JSON"));
        assert!(!is_corruption_message("malformed input"));

        // Other non-corruption
        assert!(!is_corruption_message("connection refused"));
        assert!(!is_corruption_message("timeout error"));
    }

    // =========================================================================
    // MCP Error Exit Code Tests
    // =========================================================================

    #[test]
    fn test_mcp_error_to_exit_code_server_not_running() {
        let err = McpClientError::ServerNotRunning {
            host: "127.0.0.1".to_string(),
            port: 3100,
            source: std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused"),
        };
        assert_eq!(mcp_error_to_exit_code(&err), 1);
    }

    #[test]
    fn test_mcp_error_to_exit_code_connection_timeout() {
        let err = McpClientError::ConnectionTimeout {
            host: "127.0.0.1".to_string(),
            port: 3100,
            timeout_ms: 5000,
        };
        assert_eq!(mcp_error_to_exit_code(&err), 1);
    }

    #[test]
    fn test_mcp_error_to_exit_code_corruption() {
        let err = McpClientError::McpError {
            code: -32000,
            message: "corruption detected".to_string(),
        };
        assert_eq!(mcp_error_to_exit_code(&err), 2);

        let err = McpClientError::McpError {
            code: -32000,
            message: "stale index".to_string(),
        };
        assert_eq!(mcp_error_to_exit_code(&err), 2);
    }

    #[test]
    fn test_mcp_error_to_exit_code_regular_error() {
        let err = McpClientError::McpError {
            code: -32600,
            message: "Invalid request".to_string(),
        };
        assert_eq!(mcp_error_to_exit_code(&err), 1);
    }

    #[test]
    fn test_mcp_error_to_exit_code_no_result() {
        let err = McpClientError::NoResult;
        assert_eq!(mcp_error_to_exit_code(&err), 1);
    }
}
