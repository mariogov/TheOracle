//! Edge case tests for the semantic embedding model.

use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

use super::super::SEMANTIC_DIMENSION;
use super::helpers::create_test_model;

#[tokio::test]
async fn test_edge_1_empty_input_text() {
    let model = create_test_model().await;
    model.load().await.unwrap();

    // Empty string should error on ModelInput::text()
    let result = ModelInput::text("");
    assert!(
        result.is_err(),
        "Empty string should error on ModelInput::text"
    );

    // However, if we construct text with just whitespace, it should still work
    let input = ModelInput::text(" ").expect("Whitespace input should work");
    let result = model.embed(&input).await;

    assert!(
        result.is_ok(),
        "Whitespace input should produce valid embedding"
    );
    let embedding = result.unwrap();
    assert_eq!(
        embedding.vector.len(),
        SEMANTIC_DIMENSION,
        "Vector dimension must be 1024"
    );
}
