//! Tests for ProfileManager
//!
//! Comprehensive test suite for ProfileManager functionality.

#[cfg(test)]
mod tests {
    use crate::teleological::services::profile_manager::ProfileManager;
    use crate::teleological::{ProfileId, NUM_EMBEDDERS};

    #[test]
    fn test_create_profile() {
        let mut manager = ProfileManager::new();

        let weights = [0.1; NUM_EMBEDDERS];
        let profile = manager.create_profile("test_profile", weights);

        assert_eq!(profile.id.as_str(), "test_profile");
        assert_eq!(manager.profile_count(), 4); // 3 built-in + 1 new

        // Weights should be normalized
        let sum: f32 = profile.embedding_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_get_profile() {
        let manager = ProfileManager::new();

        let code_id = ProfileId::new("code_implementation");
        let profile = manager.get_profile(&code_id);

        assert!(profile.is_some());
        assert_eq!(profile.unwrap().id, code_id);

        let non_existent = ProfileId::new("non_existent");
        assert!(manager.get_profile(&non_existent).is_none());
    }

    #[test]
    fn test_update_profile() {
        let mut manager = ProfileManager::new();

        let code_id = ProfileId::new("code_implementation");

        // Update with new weights
        let mut new_weights = [0.05; NUM_EMBEDDERS];
        new_weights[0] = 0.5; // Boost E1

        let updated = manager.update_profile(&code_id, new_weights);
        assert!(updated);

        // Verify update
        let profile = manager.get_profile(&code_id).unwrap();
        assert!(profile.embedding_weights[0] > 0.3); // Should be normalized but still high

        // Update non-existent profile
        let fake_id = ProfileId::new("fake");
        assert!(!manager.update_profile(&fake_id, [0.1; NUM_EMBEDDERS]));
    }

    #[test]
    fn test_delete_profile() {
        let mut manager = ProfileManager::new();

        let code_id = ProfileId::new("code_implementation");
        assert!(manager.contains(&code_id));

        let deleted = manager.delete_profile(&code_id);
        assert!(deleted);
        assert!(!manager.contains(&code_id));
        assert_eq!(manager.profile_count(), 2);

        // Delete non-existent
        let fake_id = ProfileId::new("fake");
        assert!(!manager.delete_profile(&fake_id));
    }

    #[test]
    fn test_list_profiles() {
        let manager = ProfileManager::new();

        let ids = manager.list_profiles();
        assert_eq!(ids.len(), 3);

        let id_strs: Vec<&str> = ids.iter().map(|id| id.as_str()).collect();
        assert!(id_strs.contains(&"code_implementation"));
        assert!(id_strs.contains(&"research_analysis"));
        assert!(id_strs.contains(&"creative_writing"));
    }

    #[test]
    fn test_find_best_match_code() {
        let manager = ProfileManager::new();

        let result = manager.find_best_match("implement a sorting algorithm");
        assert!(result.is_some());

        let matched = result.unwrap();
        assert_eq!(matched.profile_id.as_str(), "code_implementation");
        assert!(matched.similarity > 0.1);
        assert!(
            matched.reason.contains("code")
                || matched.reason.contains("implement")
                || matched.reason.contains("algorithm")
        );
    }
}
