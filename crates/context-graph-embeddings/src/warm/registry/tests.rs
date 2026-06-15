//! Basic registration tests for the warm model registry.

use super::*;
use crate::warm::error::WarmError;
use crate::warm::handle::ModelHandle;
use crate::warm::state::WarmModelState;

/// Helper to create a test ModelHandle
#[allow(dead_code)]
pub(crate) fn test_handle(bytes: usize) -> ModelHandle {
    ModelHandle::new(0x1000_0000, bytes, 0, 0xDEAD_BEEF)
}

// ==================== Basic Registration Tests ====================

#[test]
fn test_new_registry_is_empty() {
    let registry = WarmModelRegistry::new();
    assert_eq!(registry.model_count(), 0);
    assert!(!registry.all_warm()); // Empty registry is not "all warm"
    assert!(!registry.any_failed());
}

#[test]
fn test_register_model_success() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();

    assert_eq!(registry.model_count(), 1);
    let entry = registry.get_entry("E1_Semantic").unwrap();
    assert_eq!(entry.model_id, "E1_Semantic");
    assert_eq!(entry.expected_bytes, 512 * 1024 * 1024);
    assert_eq!(entry.expected_dimension, 768);
    assert!(matches!(entry.state, WarmModelState::Pending));
    assert!(entry.handle.is_none());
}

#[test]
fn test_register_model_duplicate_fails() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();

    let err = registry
        .register_model("E1_Semantic", 256 * 1024 * 1024, 512)
        .unwrap_err();

    match err {
        WarmError::ModelAlreadyRegistered { model_id } => {
            assert_eq!(model_id, "E1_Semantic");
        }
        _ => panic!("Expected ModelAlreadyRegistered error"),
    }
}

#[test]
fn test_register_all_models() {
    let mut registry = WarmModelRegistry::new();

    // Register all 12 embedding models
    for (i, model_id) in EMBEDDING_MODEL_IDS.iter().enumerate() {
        registry
            .register_model(*model_id, (i + 1) * 100 * 1024 * 1024, 768)
            .unwrap();
    }

    assert_eq!(registry.model_count(), TOTAL_MODEL_COUNT);
}
