//! Tests for WarmModelState transitions and predicates.

use crate::warm::state::WarmModelState;

#[test]
fn test_pending_predicates() {
    let s = WarmModelState::Pending;
    assert!(!s.is_warm());
    assert!(!s.is_failed());
    assert!(!s.is_loading());
}

#[test]
fn test_loading_predicates() {
    let s = WarmModelState::Loading {
        progress_percent: 50,
        bytes_loaded: 1024,
    };
    assert!(!s.is_warm());
    assert!(!s.is_failed());
    assert!(s.is_loading());
}

#[test]
fn test_validating_predicates() {
    let s = WarmModelState::Validating;
    assert!(!s.is_warm());
    assert!(!s.is_failed());
    assert!(!s.is_loading());
}

#[test]
fn test_warm_predicates() {
    let s = WarmModelState::Warm;
    assert!(s.is_warm());
    assert!(!s.is_failed());
    assert!(!s.is_loading());
}

#[test]
fn test_failed_predicates() {
    let s = WarmModelState::Failed {
        error_code: 101,
        error_message: "VRAM exhausted".into(),
    };
    assert!(!s.is_warm());
    assert!(s.is_failed());
    assert!(!s.is_loading());
}

#[test]
fn test_state_equality() {
    assert_eq!(WarmModelState::Pending, WarmModelState::Pending);
    assert_eq!(WarmModelState::Warm, WarmModelState::Warm);
    assert_ne!(WarmModelState::Pending, WarmModelState::Warm);

    let loading1 = WarmModelState::Loading {
        progress_percent: 50,
        bytes_loaded: 1024,
    };
    let loading2 = WarmModelState::Loading {
        progress_percent: 50,
        bytes_loaded: 1024,
    };
    let loading3 = WarmModelState::Loading {
        progress_percent: 75,
        bytes_loaded: 2048,
    };
    assert_eq!(loading1, loading2);
    assert_ne!(loading1, loading3);
}

#[test]
fn test_state_clone() {
    let original = WarmModelState::Failed {
        error_code: 102,
        error_message: "Load failed".to_string(),
    };
    let cloned = original.clone();

    if let (
        WarmModelState::Failed {
            error_code: c1,
            error_message: m1,
        },
        WarmModelState::Failed {
            error_code: c2,
            error_message: m2,
        },
    ) = (original, cloned)
    {
        assert_eq!(c1, c2);
        assert_eq!(m1, m2);
    } else {
        panic!("Clone should produce Failed state");
    }
}
