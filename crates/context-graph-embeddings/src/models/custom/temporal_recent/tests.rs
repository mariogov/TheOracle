//! Integration and edge case tests for the Temporal-Recent model.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use chrono::{Duration, Utc};

    use crate::models::custom::temporal_recent::{TemporalRecentModel, TEMPORAL_RECENT_DIMENSION};
    use crate::traits::EmbeddingModel;
    use crate::types::ModelInput;

    #[tokio::test]
    async fn test_recent_vs_old_timestamps() {
        let ref_time = Utc::now();
        let model = TemporalRecentModel::with_reference_time(ref_time);

        // Recent: 1 hour ago
        let recent = ref_time - Duration::hours(1);
        let recent_input = ModelInput::text_with_instruction(
            "content",
            format!("timestamp:{}", recent.to_rfc3339()),
        )
        .expect("Failed to create");

        // Somewhat old: 7 days ago (not too old to have significant decay)
        let old = ref_time - Duration::days(7);
        let old_input =
            ModelInput::text_with_instruction("content", format!("timestamp:{}", old.to_rfc3339()))
                .expect("Failed to create");

        let recent_embedding = model.embed(&recent_input).await.expect("Recent embed");
        let old_embedding = model.embed(&old_input).await.expect("Old embed");

        // Both should be valid 512D vectors
        assert_eq!(recent_embedding.vector.len(), 512);
        assert_eq!(old_embedding.vector.len(), 512);

        // Vectors should be different
        assert_ne!(
            recent_embedding.vector, old_embedding.vector,
            "Recent and old embeddings should differ"
        );

        // Both should be L2 normalized
        let recent_norm: f32 = recent_embedding
            .vector
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let old_norm: f32 = old_embedding
            .vector
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        assert!(
            (recent_norm - 1.0).abs() < 0.001,
            "Recent should be normalized"
        );
        assert!((old_norm - 1.0).abs() < 0.001, "Old should be normalized");
    }

    #[test]
    fn test_dimension_constant_matches() {
        assert_eq!(
            TEMPORAL_RECENT_DIMENSION, 512,
            "TEMPORAL_RECENT_DIMENSION must be 512"
        );
    }
}
