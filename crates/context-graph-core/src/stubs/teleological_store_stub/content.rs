//! Content storage operations for the in-memory teleological store.
//!
//! This module implements content text storage (TASK-CONTENT-004).

use tracing::debug;
use uuid::Uuid;

use super::InMemoryTeleologicalStore;
use crate::error::CoreResult;

impl InMemoryTeleologicalStore {
    /// Store content text for a fingerprint.
    pub async fn store_content_impl(&self, id: Uuid, content: &str) -> CoreResult<()> {
        self.content.insert(id, content.to_string());
        debug!(
            fingerprint_id = %id,
            content_size = content.len(),
            "Content stored"
        );
        Ok(())
    }

    /// Retrieve content text for a fingerprint.
    pub async fn get_content_impl(&self, id: Uuid) -> CoreResult<Option<String>> {
        Ok(self.content.get(&id).map(|r| r.clone()))
    }

    /// Retrieve content text for multiple fingerprints.
    pub async fn get_content_batch_impl(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>> {
        Ok(ids
            .iter()
            .map(|id| self.content.get(id).map(|r| r.clone()))
            .collect())
    }

    /// Delete content text for a fingerprint.
    pub async fn delete_content_impl(&self, id: Uuid) -> CoreResult<bool> {
        let removed = self.content.remove(&id).is_some();
        if removed {
            debug!(fingerprint_id = %id, "Content deleted");
        }
        Ok(removed)
    }
}
