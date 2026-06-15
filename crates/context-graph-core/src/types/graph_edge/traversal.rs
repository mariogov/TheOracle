//! Traversal tracking and shortcut detection methods for GraphEdge.

use chrono::Utc;

use super::edge::GraphEdge;

impl GraphEdge {
    /// Record a traversal of this edge.
    ///
    /// Updates traversal_count (saturating) and last_traversed_at timestamp.
    /// Per constitution.yaml dream.amortized: `trigger: "3+ hop path traversed ≥5×"`
    #[inline]
    pub fn record_traversal(&mut self) {
        self.traversal_count = self.traversal_count.saturating_add(1);
        self.last_traversed_at = Some(Utc::now());
    }

    /// Check if this edge is a reliable amortized shortcut.
    ///
    /// Per constitution.yaml edge_model.amortized:
    /// - `trigger: "3+ hop path traversed ≥5×"` (we check traversal_count >= 3)
    /// - `confidence: "≥0.7"`
    /// - `steering_reward > 0.3` (positive feedback)
    /// - `is_amortized_shortcut == true`
    ///
    /// # Returns
    /// `true` if ALL conditions are met:
    /// 1. `is_amortized_shortcut == true`
    /// 2. `traversal_count >= 3`
    /// 3. `steering_reward > 0.3`
    /// 4. `confidence >= 0.7`
    #[inline]
    pub fn is_reliable_shortcut(&self) -> bool {
        self.is_amortized_shortcut
            && self.traversal_count >= 3
            && self.steering_reward > 0.3
            && self.confidence >= 0.7
    }

    /// Mark this edge as an amortized shortcut.
    ///
    /// Called during dream consolidation when a 3+ hop path has been
    /// traversed enough times to warrant a direct shortcut edge.
    #[inline]
    pub fn mark_as_shortcut(&mut self) {
        self.is_amortized_shortcut = true;
    }

    /// Get the age of this edge in seconds since creation.
    ///
    /// # Returns
    /// Number of seconds since created_at. Always >= 0.
    #[inline]
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }
}
