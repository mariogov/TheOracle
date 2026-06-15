//! Serialization and deserialization tests for configuration.

use crate::config::{Config, Phase};

// =========================================================================
// TC-GHOST-006: Configuration & Infrastructure Tests - Serialization
// =========================================================================

#[test]
fn test_config_serialization_round_trip() {
    // TC-GHOST-006: Config must serialize and deserialize exactly
    let config = Config::default_config();

    // Serialize to TOML
    let toml_str = toml::to_string(&config).expect("Config must serialize to TOML");

    // Deserialize back
    let deserialized: Config =
        toml::from_str(&toml_str).expect("Config must deserialize from TOML");

    // Verify all fields match
    assert_eq!(deserialized.phase, config.phase, "Phase must match");
    assert_eq!(
        deserialized.server.name, config.server.name,
        "Server name must match"
    );
    assert_eq!(
        deserialized.server.version, config.server.version,
        "Server version must match"
    );
    assert_eq!(
        deserialized.mcp.transport, config.mcp.transport,
        "MCP transport must match"
    );
    assert_eq!(
        deserialized.mcp.max_payload_size, config.mcp.max_payload_size,
        "MCP max_payload_size must match"
    );
    assert_eq!(
        deserialized.logging.level, config.logging.level,
        "Logging level must match"
    );
    assert_eq!(
        deserialized.embedding.model, config.embedding.model,
        "Embedding model must match"
    );
}

#[test]
fn test_config_serialization_json_round_trip() {
    // TC-GHOST-006: Config must also round-trip through JSON
    let config = Config::default_config();

    // Serialize to JSON
    let json_str = serde_json::to_string(&config).expect("Config must serialize to JSON");

    // Deserialize back
    let deserialized: Config =
        serde_json::from_str(&json_str).expect("Config must deserialize from JSON");

    // Verify critical fields
    assert_eq!(deserialized.phase, config.phase);
    assert_eq!(deserialized.embedding.model, config.embedding.model);
    assert_eq!(deserialized.cuda.device_id, config.cuda.device_id);
}

#[test]
fn test_config_phase_serialization() {
    // TC-GHOST-006: Phase enum must serialize correctly
    let phases = [Phase::Ghost, Phase::Development, Phase::Production];
    let expected = ["ghost", "development", "production"];

    for (phase, expected_str) in phases.iter().zip(expected.iter()) {
        let json = serde_json::to_string(phase).expect("Phase must serialize");
        assert_eq!(
            json,
            format!("\"{}\"", expected_str),
            "Phase {:?} must serialize as {}",
            phase,
            expected_str
        );

        let deserialized: Phase = serde_json::from_str(&json).expect("Phase must deserialize");
        assert_eq!(&deserialized, phase, "Phase must round-trip correctly");
    }
}

#[test]
fn test_config_from_toml_string() {
    // TC-GHOST-006: Config must parse from minimal TOML
    let toml_str = r#"
        [server]
        name = "test-server"
        version = "1.0.0"

        [mcp]
        transport = "stdio"
        max_payload_size = 1000000
        request_timeout = 60

        [logging]
        level = "debug"
        format = "json"
        include_location = true

        [storage]
        backend = "memory"
        path = "/tmp/test"
        compression = false

        [embedding]
        model = "custom"

        [index]
        backend = "memory"
        hnsw_m = 32
        hnsw_ef_construction = 400

        [utl]
        mode = "real"
        consolidation_threshold = 0.8

        [cuda]
        device_id = 1
        memory_limit_gb = 8.0
    "#;

    let config: Config = toml::from_str(toml_str).expect("Config must parse from TOML");

    // Verify parsed values
    assert_eq!(config.server.name, "test-server");
    assert_eq!(config.server.version, "1.0.0");
    assert_eq!(config.mcp.max_payload_size, 1000000);
    assert_eq!(config.logging.level, "debug");
    assert_eq!(config.embedding.model, "custom");
    assert_eq!(config.utl.consolidation_threshold, 0.8);
    assert_eq!(config.cuda.device_id, 1);
}

#[test]
fn test_config_full_structure_integrity() {
    // TC-GHOST-006: Full config structure must be maintained through serialization
    let mut config = Config::default_config();

    // Modify various fields
    config.server.name = "modified-server".to_string();
    config.mcp.request_timeout = 120;
    config.logging.level = "trace".to_string();
    config.embedding.model = "multi_array_13".to_string();
    config.cuda.device_id = 2;
    config.cuda.memory_limit_gb = 16.0;

    // Round-trip through TOML
    let toml_str = toml::to_string(&config).unwrap();
    let restored: Config = toml::from_str(&toml_str).unwrap();

    // Verify all modifications survived
    assert_eq!(restored.server.name, "modified-server");
    assert_eq!(restored.mcp.request_timeout, 120);
    assert_eq!(restored.logging.level, "trace");
    assert_eq!(restored.embedding.model, "multi_array_13");
    assert_eq!(restored.cuda.device_id, 2);
    assert_eq!(restored.cuda.memory_limit_gb, 16.0);
}
