//! Error handling tests for the warm model registry.

use std::sync::{Arc, RwLock};

use super::*;
use crate::warm::error::WarmError;
use crate::warm::handle::ModelHandle;
use crate::warm::state::WarmModelState;

/// Helper to create a test ModelHandle
fn test_handle(bytes: usize) -> ModelHandle {
    ModelHandle::new(0x1000_0000, bytes, 0, 0xDEAD_BEEF)
}

// ==================== Error Cases Tests ====================

#[test]
fn test_operations_on_unregistered_model() {
    let mut registry = WarmModelRegistry::new();

    // All operations should fail with ModelNotRegistered
    let err = registry.start_loading("NonExistent").unwrap_err();
    assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

    let err = registry
        .update_progress("NonExistent", 50, 1000)
        .unwrap_err();
    assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

    let err = registry.mark_validating("NonExistent").unwrap_err();
    assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

    let err = registry
        .mark_warm("NonExistent", test_handle(1000))
        .unwrap_err();
    assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

    let err = registry
        .mark_failed("NonExistent", 102, "Error")
        .unwrap_err();
    assert!(matches!(err, WarmError::ModelNotRegistered { .. }));
}

#[test]
fn test_update_progress_clamps_percent() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();
    registry.start_loading("E1_Semantic").unwrap();

    // Progress above 100 should be clamped
    registry.update_progress("E1_Semantic", 150, 1000).unwrap();

    match registry.get_state("E1_Semantic") {
        Some(WarmModelState::Loading {
            progress_percent, ..
        }) => {
            assert_eq!(progress_percent, 100);
        }
        _ => panic!("Expected Loading state"),
    }
}

// ==================== Thread Safety Documentation Test ====================

#[test]
fn test_shared_registry_type_alias() {
    // Verify SharedWarmRegistry can be created and used
    let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));

    // Write access
    {
        let mut reg = registry.write().unwrap();
        reg.register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
    }

    // Read access (concurrent readers are possible)
    {
        let reg = registry.read().unwrap();
        assert_eq!(reg.model_count(), 1);
    }

    // Verify Arc cloning works
    let registry_clone = Arc::clone(&registry);
    {
        let reg = registry_clone.read().unwrap();
        assert!(reg.get_state("E1_Semantic").is_some());
    }
}
