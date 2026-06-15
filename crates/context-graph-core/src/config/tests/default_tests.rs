//! Tests for default configuration values.

use crate::config::{
    Config, CudaConfig, EmbeddingConfig, LoggingConfig, McpConfig, ServerConfig, StorageConfig,
    UtlConfig,
};

#[test]
fn test_default_config() {
    let config = Config::default_config();
    assert_eq!(config.server.name, "context-graph");
    assert_eq!(config.cuda.device_id, 0);
}

// =========================================================================
// TC-GHOST-006: Configuration & Infrastructure Tests - Default Values
// =========================================================================

#[test]
fn test_server_config_defaults() {
    // TC-GHOST-006: Server config must have correct defaults
    let server = ServerConfig::default();

    assert_eq!(
        server.name, "context-graph",
        "Server name must be context-graph"
    );
    assert_eq!(
        server.version, "0.1.0-ghost",
        "Server version must be 0.1.0-ghost"
    );
}

#[test]
fn test_mcp_config_defaults() {
    // TC-GHOST-006: MCP config must have correct defaults
    let mcp = McpConfig::default();

    assert_eq!(mcp.transport, "stdio", "Transport must be stdio");
    assert_eq!(mcp.max_payload_size, 10_485_760, "Max payload must be 10MB");
    assert_eq!(
        mcp.request_timeout, 30,
        "Request timeout must be 30 seconds"
    );
}

#[test]
fn test_logging_config_defaults() {
    // TC-GHOST-006: Logging config must have correct defaults
    let logging = LoggingConfig::default();

    assert_eq!(logging.level, "info", "Default log level must be info");
    assert_eq!(
        logging.format, "pretty",
        "Default log format must be pretty"
    );
    assert!(
        !logging.include_location,
        "Location should be disabled by default"
    );
}

#[test]
fn test_storage_config_defaults() {
    // TC-GHOST-006: Storage config must have correct defaults
    let storage = StorageConfig::default();

    assert_eq!(storage.backend, "memory", "Default backend must be memory");
    assert_eq!(
        storage.path, "./data/storage",
        "Default path must be ./data/storage"
    );
    assert!(
        storage.compression,
        "Compression should be enabled by default"
    );
}

#[test]
fn test_embedding_config_defaults() {
    // TC-GHOST-006: Embedding config must have correct defaults
    let embedding = EmbeddingConfig::default();

    assert_eq!(embedding.model, "stub", "Default model must be stub");
}

#[test]
fn test_utl_config_defaults() {
    // TC-GHOST-006: UTL config must have correct defaults
    let utl = UtlConfig::default();

    assert_eq!(utl.mode, "stub", "Default mode must be stub");
    assert_eq!(
        utl.consolidation_threshold, 0.7,
        "Consolidation threshold must be 0.7"
    );
}

#[test]
fn test_cuda_config_defaults() {
    // TC-GHOST-006: CUDA config must have correct defaults
    let cuda = CudaConfig::default();

    assert_eq!(cuda.device_id, 0, "Default device ID must be 0");
    assert_eq!(
        cuda.memory_limit_gb, 4.0,
        "Default memory limit must be 4GB"
    );
}
