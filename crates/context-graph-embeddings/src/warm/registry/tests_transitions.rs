//! State transition tests for the warm model registry.

use super::*;
use crate::warm::error::WarmError;
use crate::warm::handle::ModelHandle;
use crate::warm::state::WarmModelState;

/// Helper to create a test ModelHandle
fn test_handle(bytes: usize) -> ModelHandle {
    ModelHandle::new(0x1000_0000, bytes, 0, 0xDEAD_BEEF)
}

#[test]
fn test_valid_state_transitions_pending_to_warm() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();

    // Pending -> Loading
    assert!(matches!(
        registry.get_state("E1_Semantic"),
        Some(WarmModelState::Pending)
    ));
    registry.start_loading("E1_Semantic").unwrap();
    assert!(matches!(
        registry.get_state("E1_Semantic"),
        Some(WarmModelState::Loading {
            progress_percent: 0,
            bytes_loaded: 0
        })
    ));

    // Update progress
    registry
        .update_progress("E1_Semantic", 50, 256 * 1024 * 1024)
        .unwrap();
    assert!(matches!(
        registry.get_state("E1_Semantic"),
        Some(WarmModelState::Loading {
            progress_percent: 50,
            bytes_loaded: _
        })
    ));

    // Loading -> Validating
    registry.mark_validating("E1_Semantic").unwrap();
    assert!(matches!(
        registry.get_state("E1_Semantic"),
        Some(WarmModelState::Validating)
    ));

    // Validating -> Warm
    registry
        .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
        .unwrap();
    assert!(matches!(
        registry.get_state("E1_Semantic"),
        Some(WarmModelState::Warm)
    ));
    assert!(registry.get_handle("E1_Semantic").is_some());
}

#[test]
fn test_invalid_transition_warm_to_loading() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();

    // Complete the full cycle to Warm
    registry.start_loading("E1_Semantic").unwrap();
    registry.mark_validating("E1_Semantic").unwrap();
    registry
        .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
        .unwrap();

    // Try to go back to Loading - should fail
    let err = registry.start_loading("E1_Semantic").unwrap_err();
    match err {
        WarmError::ModelLoadFailed {
            model_id, reason, ..
        } => {
            assert_eq!(model_id, "E1_Semantic");
            assert!(reason.contains("Invalid state transition"));
            assert!(reason.contains("Warm"));
        }
        _ => panic!("Expected ModelLoadFailed error"),
    }
}

#[test]
fn test_invalid_transition_pending_to_validating() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();

    // Cannot go directly from Pending to Validating
    let err = registry.mark_validating("E1_Semantic").unwrap_err();
    match err {
        WarmError::ModelValidationFailed {
            model_id, reason, ..
        } => {
            assert_eq!(model_id, "E1_Semantic");
            assert!(reason.contains("Invalid state transition"));
            assert!(reason.contains("Pending"));
        }
        _ => panic!("Expected ModelValidationFailed error"),
    }
}

#[test]
fn test_invalid_transition_pending_to_warm() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();

    // Cannot go directly from Pending to Warm
    let err = registry
        .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
        .unwrap_err();
    match err {
        WarmError::ModelValidationFailed { model_id, .. } => {
            assert_eq!(model_id, "E1_Semantic");
        }
        _ => panic!("Expected ModelValidationFailed error"),
    }
}

#[test]
fn test_invalid_transition_loading_to_warm() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();
    registry.start_loading("E1_Semantic").unwrap();

    // Cannot go directly from Loading to Warm (must go through Validating)
    let err = registry
        .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
        .unwrap_err();
    match err {
        WarmError::ModelValidationFailed { model_id, .. } => {
            assert_eq!(model_id, "E1_Semantic");
        }
        _ => panic!("Expected ModelValidationFailed error"),
    }
}

#[test]
fn test_mark_failed_from_loading() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();
    registry.start_loading("E1_Semantic").unwrap();

    registry
        .mark_failed("E1_Semantic", 102, "CUDA allocation failed")
        .unwrap();

    match registry.get_state("E1_Semantic") {
        Some(WarmModelState::Failed {
            error_code,
            error_message,
        }) => {
            assert_eq!(error_code, 102);
            assert_eq!(error_message, "CUDA allocation failed");
        }
        _ => panic!("Expected Failed state"),
    }
    assert!(registry.get_handle("E1_Semantic").is_none());
}

#[test]
fn test_mark_failed_from_validating() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();
    registry.start_loading("E1_Semantic").unwrap();
    registry.mark_validating("E1_Semantic").unwrap();

    registry
        .mark_failed("E1_Semantic", 103, "NaN detected in output")
        .unwrap();

    match registry.get_state("E1_Semantic") {
        Some(WarmModelState::Failed {
            error_code,
            error_message,
        }) => {
            assert_eq!(error_code, 103);
            assert_eq!(error_message, "NaN detected in output");
        }
        _ => panic!("Expected Failed state"),
    }
}

#[test]
fn test_mark_failed_from_pending_fails() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();

    // Cannot fail from Pending (must start loading first)
    let err = registry
        .mark_failed("E1_Semantic", 102, "Some error")
        .unwrap_err();
    match err {
        WarmError::ModelLoadFailed {
            model_id, reason, ..
        } => {
            assert_eq!(model_id, "E1_Semantic");
            assert!(reason.contains("Pending"));
        }
        _ => panic!("Expected ModelLoadFailed error"),
    }
}

#[test]
fn test_mark_failed_from_warm_fails() {
    let mut registry = WarmModelRegistry::new();
    registry
        .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
        .unwrap();
    registry.start_loading("E1_Semantic").unwrap();
    registry.mark_validating("E1_Semantic").unwrap();
    registry
        .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
        .unwrap();

    // Cannot fail from Warm (model is already successfully loaded)
    let err = registry
        .mark_failed("E1_Semantic", 109, "Context lost")
        .unwrap_err();
    match err {
        WarmError::ModelLoadFailed {
            model_id, reason, ..
        } => {
            assert_eq!(model_id, "E1_Semantic");
            assert!(reason.contains("Warm"));
        }
        _ => panic!("Expected ModelLoadFailed error"),
    }
}
