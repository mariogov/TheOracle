//! Concurrency tests for ModelRegistry.

use std::sync::Arc;

use crate::types::ModelId;

use super::config::ModelRegistryConfig;
use super::core::ModelRegistry;
use super::tests::TestFactory;

#[tokio::test]
async fn test_concurrent_get_same_model_loads_once() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = Arc::new(ModelRegistry::new(config, factory.clone()).await.unwrap());

    // Spawn 100 concurrent get_model calls
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let r = Arc::clone(&registry);
            tokio::spawn(async move { r.get_model(ModelId::Semantic).await })
        })
        .collect();

    // Wait for all to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // Should only have created the model once
    assert_eq!(factory.create_count(), 1);

    let stats = registry.stats().await;
    assert_eq!(stats.load_count, 1);
    // 99 cache hits (first one triggers load)
    assert_eq!(stats.cache_hits, 99);
}

#[tokio::test]
async fn test_concurrent_load_different_models() {
    let factory = Arc::new(TestFactory::new());
    let config = ModelRegistryConfig::default();

    let registry = Arc::new(ModelRegistry::new(config, factory).await.unwrap());

    let models = [
        ModelId::Semantic,
        ModelId::Code,
        ModelId::Graph,
        ModelId::Entity,
        ModelId::Hdc,
    ];

    let handles: Vec<_> = models
        .iter()
        .map(|model_id| {
            let r = Arc::clone(&registry);
            let mid = *model_id;
            tokio::spawn(async move { r.load_model(mid).await })
        })
        .collect();

    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    assert_eq!(registry.loaded_count().await, 5);
}
