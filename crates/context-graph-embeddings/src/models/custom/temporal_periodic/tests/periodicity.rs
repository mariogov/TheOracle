//! Periodicity and Fourier property tests for TemporalPeriodicModel.

use crate::models::custom::temporal_periodic::TemporalPeriodicModel;
use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

#[tokio::test]
async fn test_deterministic_with_same_timestamp() {
    let model = TemporalPeriodicModel::new();

    let timestamp = "timestamp:2024-01-15T10:30:00Z";
    let input1 = ModelInput::text_with_instruction("content", timestamp).expect("Failed to create");
    let input2 = ModelInput::text_with_instruction("content", timestamp).expect("Failed to create");

    let embedding1 = model.embed(&input1).await.expect("First embed");
    let embedding2 = model.embed(&input2).await.expect("Second embed");

    assert_eq!(
        embedding1.vector, embedding2.vector,
        "Same timestamp must produce identical embeddings"
    );
}
