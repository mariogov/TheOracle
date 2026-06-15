//! Token embedding and pooling tests for LateInteractionModel.

use super::tests::create_and_load_model;

#[tokio::test]
async fn test_embed_tokens_produces_per_token_vectors() {
    let model = create_and_load_model().await;
    let tokens = model
        .embed_tokens("hello world test")
        .await
        .expect("embed_tokens");
    assert!(tokens.vectors.len() >= 3);
}
