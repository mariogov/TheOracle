//! Integration tests for hybrid E4 session+position encoding mode.

use crate::models::custom::temporal_positional::TemporalPositionalModel;
use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

/// Create a text input with instruction.
fn text_input(instruction: &str) -> ModelInput {
    ModelInput::text_with_instruction("test", instruction).expect("Failed to create input")
}

#[tokio::test]
async fn test_hybrid_embedding_dimension() {
    let model = TemporalPositionalModel::new();
    assert!(model.is_hybrid_mode(), "Default should be hybrid mode");

    let input = text_input("session:abc123 sequence:42");
    let emb = model.embed(&input).await.unwrap();

    assert_eq!(emb.vector.len(), 512, "Hybrid embedding should be 512D");
}
