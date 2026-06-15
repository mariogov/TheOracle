//! Basic tests for the semantic embedding model.

use crate::traits::EmbeddingModel;
use crate::types::{InputType, ModelId};

use super::helpers::create_test_model;

#[tokio::test]
async fn test_model_id_is_semantic() {
    let model = create_test_model().await;
    assert_eq!(model.model_id(), ModelId::Semantic);
}

#[tokio::test]
async fn test_supported_input_types_is_text() {
    let model = create_test_model().await;
    let types = model.supported_input_types();
    assert_eq!(types.len(), 1);
    assert_eq!(types[0], InputType::Text);
}
