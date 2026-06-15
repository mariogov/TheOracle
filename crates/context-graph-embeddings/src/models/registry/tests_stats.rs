//! Statistics and edge case tests for ModelRegistry.

use std::sync::Arc;

use crate::types::ModelId;

use super::config::ModelRegistryConfig;
use super::core::ModelRegistry;
use super::tests::TestFactory;

#[tokio::test]
async fn test_stats_initial_zeros() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();
    let stats = registry.stats().await;

    assert_eq!(stats.loaded_count, 0);
    assert_eq!(stats.total_memory_bytes, 0);
    assert_eq!(stats.load_count, 0);
    assert_eq!(stats.unload_count, 0);
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.load_failures, 0);
}

#[tokio::test]
async fn test_stats_accurate_after_operations() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = ModelRegistry::new(config, factory).await.unwrap();

    // Load 3 models
    registry.load_model(ModelId::Semantic).await.unwrap();
    registry.load_model(ModelId::Code).await.unwrap();
    registry.load_model(ModelId::Graph).await.unwrap();

    // Unload 1
    registry.unload_model(ModelId::Code).await.unwrap();

    // Get (cache hit)
    let _ = registry.get_model(ModelId::Semantic).await.unwrap();

    let stats = registry.stats().await;
    assert_eq!(stats.load_count, 3);
    assert_eq!(stats.unload_count, 1);
    assert_eq!(stats.loaded_count, 2);
    assert_eq!(stats.cache_hits, 1);
}
