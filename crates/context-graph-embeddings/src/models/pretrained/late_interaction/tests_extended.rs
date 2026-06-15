//! Extended and edge case tests for LateInteractionModel.

use super::tests::create_and_load_model;
use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

#[tokio::test]
async fn test_embed_returns_128d() {
    let model = create_and_load_model().await;
    let input = ModelInput::text("ColBERT test").expect("Input");
    let embedding = model.embed(&input).await.expect("Embed should succeed");
    assert_eq!(embedding.vector.len(), 128);
}
