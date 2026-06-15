//! Tests for error module.

use super::*;
use crate::teleological::embedder::Embedder;
use uuid::Uuid;

// ========== ContextGraphError Tests ==========

#[test]
fn test_context_graph_error_codes() {
    // Test all error variants have correct codes
    let e = ContextGraphError::Embedding(EmbeddingError::EmptyInput);
    assert_eq!(e.error_code(), -32005);

    let e = ContextGraphError::Storage(StorageError::Database("test".to_string()));
    assert_eq!(e.error_code(), -32004);

    let e = ContextGraphError::Index(IndexError::Hnsw("test".to_string()));
    assert_eq!(e.error_code(), -32008);

    let e = ContextGraphError::Config(ConfigError::Missing("test".to_string()));
    assert_eq!(e.error_code(), -32603);

    let e = ContextGraphError::Gpu(GpuError::NotAvailable);
    assert_eq!(e.error_code(), -32009);

    let e = ContextGraphError::Mcp(McpError::InvalidParams("test".to_string()));
    assert_eq!(e.error_code(), -32602);

    let e = ContextGraphError::Validation("test".to_string());
    assert_eq!(e.error_code(), -32602);

    let e = ContextGraphError::Internal("test".to_string());
    assert_eq!(e.error_code(), -32603);

    println!("[PASS] All error codes mapped correctly");
}

#[test]
fn test_is_recoverable() {
    // Recoverable errors
    let e = ContextGraphError::Embedding(EmbeddingError::ModelNotLoaded(Embedder::Semantic));
    assert!(e.is_recoverable());

    let e = ContextGraphError::Storage(StorageError::Transaction("test".to_string()));
    assert!(e.is_recoverable());

    let e = ContextGraphError::Index(IndexError::Timeout(1000));
    assert!(e.is_recoverable());

    let e = ContextGraphError::Mcp(McpError::RateLimited("429".to_string()));
    assert!(e.is_recoverable());

    let e = ContextGraphError::Gpu(GpuError::OutOfMemory {
        requested: 1000,
        available: 500,
    });
    assert!(e.is_recoverable());

    // Non-recoverable errors
    let e = ContextGraphError::Validation("bad".to_string());
    assert!(!e.is_recoverable());

    let e = ContextGraphError::Storage(StorageError::Corruption("bad".to_string()));
    assert!(!e.is_recoverable());

    println!("[PASS] is_recoverable() works correctly");
}

#[test]
fn test_is_critical() {
    // Critical errors
    let e = ContextGraphError::Storage(StorageError::Corruption("bad".to_string()));
    assert!(e.is_critical());

    let e = ContextGraphError::Index(IndexError::Corruption(
        Embedder::Semantic,
        "checksum".to_string(),
    ));
    assert!(e.is_critical());

    let e = ContextGraphError::Gpu(GpuError::NotAvailable);
    assert!(e.is_critical());

    let e = ContextGraphError::Internal("bug".to_string());
    assert!(e.is_critical());

    // Non-critical errors
    let e = ContextGraphError::Validation("bad".to_string());
    assert!(!e.is_critical());

    let e = ContextGraphError::Embedding(EmbeddingError::EmptyInput);
    assert!(!e.is_critical());

    println!("[PASS] is_critical() works correctly");
}

#[test]
fn test_mcp_error_codes() {
    assert_eq!(
        McpError::InvalidRequest("".to_string()).error_code(),
        -32600
    );
    assert_eq!(
        McpError::MethodNotFound("".to_string()).error_code(),
        -32601
    );
    assert_eq!(McpError::InvalidParams("".to_string()).error_code(), -32602);
    assert_eq!(McpError::RateLimited("".to_string()).error_code(), -32005);
    assert_eq!(McpError::Unauthorized("".to_string()).error_code(), -32006);
    assert_eq!(McpError::SessionExpired.error_code(), -32000);
    assert_eq!(McpError::PiiDetected.error_code(), -32007);

    println!("[PASS] McpError codes correct");
}

// ========== Edge Case Tests ==========

#[test]
fn edge_case_empty_validation_message() {
    println!("=== BEFORE: Empty validation message ===");
    let e = ContextGraphError::Validation("".to_string());
    println!("Error: {}", e);
    println!("Code: {}", e.error_code());
    println!("=== AFTER: Should display 'Validation error: ' ===");
    assert!(e.to_string().contains("Validation"));
    println!("[PASS] Empty message handled");
}

#[test]
fn edge_case_unicode_in_error() {
    println!("=== BEFORE: Unicode in error message ===");
    let e = ContextGraphError::Internal("错误消息 🔥".to_string());
    println!("Error: {}", e);
    println!("=== AFTER: Unicode should be preserved ===");
    assert!(e.to_string().contains("🔥"));
    assert!(e.to_string().contains("错误"));
    println!("[PASS] Unicode preserved");
}

#[test]
fn edge_case_nested_from_conversion() {
    println!("=== BEFORE: From conversion chain ===");
    let storage_err = StorageError::Database("connection lost".to_string());
    let ctx_err: ContextGraphError = storage_err.into();
    println!("Original: connection lost");
    println!("Converted: {}", ctx_err);
    println!("=== AFTER: Message should contain 'connection lost' ===");
    assert!(ctx_err.to_string().contains("connection lost"));
    println!("[PASS] From conversion preserves message");
}

#[test]
fn edge_case_all_embedders_in_errors() {
    println!("=== Testing all 13 embedders in errors ===");
    for embedder in Embedder::all() {
        let e = EmbeddingError::ModelNotLoaded(embedder);
        assert!(e.to_string().contains(&format!("{:?}", embedder)));
    }
    println!("[PASS] All 13 embedders work in errors");
}

#[test]
fn edge_case_zero_values() {
    // Test with zero/nil values
    let e = ContextGraphError::Storage(StorageError::NotFound(Uuid::nil()));
    assert!(e.to_string().contains("00000000"));

    let e = ContextGraphError::Index(IndexError::Timeout(0));
    assert!(e.to_string().contains("0ms"));

    let e = ContextGraphError::Gpu(GpuError::OutOfMemory {
        requested: 0,
        available: 0,
    });
    assert!(e.to_string().contains("0 bytes"));

    println!("[PASS] Zero values handled");
}

#[test]
fn test_convenience_constructors() {
    let e = ContextGraphError::internal("bug found");
    assert!(matches!(e, ContextGraphError::Internal(_)));
    assert!(e.to_string().contains("bug found"));

    let e = ContextGraphError::validation("bad input");
    assert!(matches!(e, ContextGraphError::Validation(_)));
    assert!(e.to_string().contains("bad input"));

    println!("[PASS] Convenience constructors work");
}

// ========== Legacy CoreError Tests ==========

#[test]
fn test_core_error_display() {
    let err = CoreError::NodeNotFound { id: Uuid::nil() };
    assert!(err.to_string().contains("Node not found"));
}

#[test]
fn test_core_error_dimension_mismatch() {
    let err = CoreError::DimensionMismatch {
        expected: 1536,
        actual: 768,
    };
    assert!(err.to_string().contains("1536"));
    assert!(err.to_string().contains("768"));
}

#[test]
fn test_core_to_context_graph_conversion() {
    let core_err = CoreError::StorageError("db failed".to_string());
    let ctx_err: ContextGraphError = core_err.into();
    assert!(matches!(ctx_err, ContextGraphError::Storage(_)));
    assert!(ctx_err.to_string().contains("db failed"));

    println!("[PASS] CoreError converts to ContextGraphError");
}
