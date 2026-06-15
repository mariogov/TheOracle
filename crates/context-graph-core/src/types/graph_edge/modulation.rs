//! Steering modulation methods for GraphEdge.

use super::edge::GraphEdge;

impl GraphEdge {
    /// Computes the modulated weight considering steering reward.
    ///
    /// This is the primary method for getting an edge's effective weight during
    /// graph traversal. It applies steering reward feedback to the base weight.
    ///
    /// # Formula
    ///
    /// ```text
    /// modulated = (weight * (1.0 + steering_reward * 0.2)).clamp(0.0, 1.0)
    /// ```
    ///
    /// # Returns
    ///
    /// Effective weight in [0.0, 1.0], never NaN or Infinity (per AP-009).
    #[inline]
    pub fn get_modulated_weight(&self) -> f32 {
        (self.weight * (1.0 + self.steering_reward * 0.2)).clamp(0.0, 1.0)
    }

    /// Applies a steering reward signal from the Steering Subsystem.
    ///
    /// The steering reward provides reinforcement learning feedback:
    /// - Positive rewards strengthen the edge (encourage traversal)
    /// - Negative rewards weaken the edge (discourage traversal)
    ///
    /// Rewards are accumulated (additive) and clamped to [-1.0, 1.0].
    ///
    /// # Arguments
    ///
    /// * `reward` - Reward signal to add. Positive reinforces, negative discourages.
    #[inline]
    pub fn apply_steering_reward(&mut self, reward: f32) {
        self.steering_reward = (self.steering_reward + reward).clamp(-1.0, 1.0);
    }

    /// Decay the steering reward by a factor.
    ///
    /// Used to gradually reduce influence of old rewards over time.
    /// Does NOT clamp - assumes decay_factor is in [0.0, 1.0].
    ///
    /// # Arguments
    /// * `decay_factor` - Multiplicative decay (e.g., 0.9 reduces by 10%)
    #[inline]
    pub fn decay_steering(&mut self, decay_factor: f32) {
        self.steering_reward *= decay_factor;
    }
}
