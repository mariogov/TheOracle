//! Construction and configuration tests for TemporalPositionalModel.

use crate::traits::EmbeddingModel;

use super::super::{TemporalPositionalModel, DEFAULT_BASE};

#[test]
fn test_new_creates_initialized_model() {
    let model = TemporalPositionalModel::new();

    println!("BEFORE: model created");
    println!("AFTER: is_initialized = {}", model.is_initialized());

    assert!(
        model.is_initialized(),
        "Custom model must be initialized immediately"
    );
    assert_eq!(model.base(), DEFAULT_BASE, "Must use default base");
}
