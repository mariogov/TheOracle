//! Manual output verification tests for ModelRegistry.

use std::sync::Arc;

use crate::traits::ModelFactory;
use crate::types::{ModelId, ModelInput};

use super::config::ModelRegistryConfig;
use super::core::ModelRegistry;
use super::tests::TestFactory;

#[tokio::test]
async fn test_mov_1_model_instance_valid() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    registry.load_model(ModelId::Semantic).await.unwrap();

    let model = registry.get_model(ModelId::Semantic).await.unwrap();

    // Verify Arc has at least 2 refs (registry + this)
    assert!(Arc::strong_count(&model) >= 2);

    // Verify model is functional
    let input = ModelInput::text("Test").unwrap();
    let embedding = model.embed(&input).await.unwrap();
    assert_eq!(embedding.model_id, ModelId::Semantic);
}

#[tokio::test]
async fn test_mov_2_memory_tracker_consistency() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory.clone()).await.unwrap();

    // Load several models
    registry.load_model(ModelId::Semantic).await.unwrap();
    registry.load_model(ModelId::Code).await.unwrap();
    registry.load_model(ModelId::Graph).await.unwrap();

    // Calculate expected from factory estimates
    let loaded = registry.loaded_models().await;
    let expected: usize = loaded.iter().map(|id| factory.estimate_memory(*id)).sum();

    let actual = registry.total_memory_usage().await;

    // Verify within 1% tolerance
    let diff = expected.abs_diff(actual);
    let tolerance = expected / 100;
    assert!(
        diff <= tolerance,
        "Memory mismatch: expected {}, actual {}, diff {}, tolerance {}",
        expected,
        actual,
        diff,
        tolerance
    );
}
