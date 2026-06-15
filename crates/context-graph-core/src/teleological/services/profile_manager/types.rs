//! Types for ProfileManager
//!
//! Contains configuration, result, and internal types used by ProfileManager.

use crate::teleological::ProfileId;

/// Configuration for ProfileManager.
#[derive(Clone, Debug)]
pub struct ProfileManagerConfig {
    /// Maximum number of profiles to store.
    pub max_profiles: usize,
    /// Automatically create default profile if none exists.
    pub auto_create: bool,
    /// Default profile ID when no match is found.
    pub default_profile_id: String,
}

impl Default for ProfileManagerConfig {
    fn default() -> Self {
        Self {
            max_profiles: 100,
            auto_create: true,
            default_profile_id: "code_implementation".to_string(),
        }
    }
}

/// Result of profile matching.
#[derive(Clone, Debug)]
pub struct ProfileMatch {
    /// The matched profile ID.
    pub profile_id: ProfileId,
    /// Similarity score between context and profile.
    pub similarity: f32,
    /// Reason for the match.
    pub reason: String,
}

/// Usage statistics for a profile.
#[derive(Clone, Debug)]
pub struct ProfileStats {
    /// Profile identifier.
    pub profile_id: ProfileId,
    /// Total number of times this profile was used.
    pub usage_count: usize,
    /// Average effectiveness score from recorded usages.
    pub avg_effectiveness: f32,
    /// Timestamp of last usage (epoch millis).
    pub last_used: u64,
}

/// Internal stats tracking.
#[derive(Clone, Debug, Default)]
pub(crate) struct InternalStats {
    pub(crate) usage_count: usize,
    pub(crate) total_effectiveness: f32,
    pub(crate) last_used: u64,
}
