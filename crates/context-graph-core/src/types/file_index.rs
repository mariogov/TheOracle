//! File index types for file watcher management.
//!
//! These types support the file index that maps file paths to fingerprint IDs,
//! enabling efficient lookup, cleanup, and reconciliation of MDFileChunk embeddings.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Entry in the file index mapping file paths to fingerprint IDs.
///
/// Stores the list of fingerprint UUIDs for a given file path along with
/// metadata about when the index was last updated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileIndexEntry {
    /// The file path this entry represents.
    pub file_path: String,
    /// UUIDs of fingerprints (chunks) for this file.
    pub fingerprint_ids: Vec<Uuid>,
    /// When this entry was last updated.
    pub last_updated: DateTime<Utc>,
}

impl FileIndexEntry {
    /// Create a new file index entry.
    pub fn new(file_path: String) -> Self {
        Self {
            file_path,
            fingerprint_ids: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    /// Add a fingerprint ID to this entry.
    pub fn add_fingerprint(&mut self, id: Uuid) {
        if !self.fingerprint_ids.contains(&id) {
            self.fingerprint_ids.push(id);
            self.last_updated = Utc::now();
        }
    }

    /// Remove a fingerprint ID from this entry.
    /// Returns true if the ID was found and removed.
    pub fn remove_fingerprint(&mut self, id: Uuid) -> bool {
        if let Some(pos) = self.fingerprint_ids.iter().position(|&x| x == id) {
            self.fingerprint_ids.remove(pos);
            self.last_updated = Utc::now();
            true
        } else {
            false
        }
    }

    /// Get the number of fingerprints for this file.
    pub fn fingerprint_count(&self) -> usize {
        self.fingerprint_ids.len()
    }

    /// Check if this entry is empty (no fingerprints).
    pub fn is_empty(&self) -> bool {
        self.fingerprint_ids.is_empty()
    }
}

/// Statistics about file watcher content in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWatcherStats {
    /// Total number of files with embeddings.
    pub total_files: usize,
    /// Total number of fingerprints (chunks) across all files.
    pub total_chunks: usize,
    /// Average chunks per file.
    pub avg_chunks_per_file: f64,
    /// Minimum chunks in any file.
    pub min_chunks: usize,
    /// Maximum chunks in any file.
    pub max_chunks: usize,
}

impl Default for FileWatcherStats {
    fn default() -> Self {
        Self {
            total_files: 0,
            total_chunks: 0,
            avg_chunks_per_file: 0.0,
            min_chunks: 0,
            max_chunks: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_index_entry_new() {
        let entry = FileIndexEntry::new("/test/path.md".to_string());
        assert_eq!(entry.file_path, "/test/path.md");
        assert!(entry.fingerprint_ids.is_empty());
        assert!(entry.is_empty());
    }

    #[test]
    fn test_file_index_entry_add_fingerprint() {
        let mut entry = FileIndexEntry::new("/test/path.md".to_string());
        let id = Uuid::new_v4();

        entry.add_fingerprint(id);
        assert_eq!(entry.fingerprint_count(), 1);
        assert!(!entry.is_empty());
        assert!(entry.fingerprint_ids.contains(&id));

        // Adding same ID again should not duplicate
        entry.add_fingerprint(id);
        assert_eq!(entry.fingerprint_count(), 1);
    }

    #[test]
    fn test_file_index_entry_remove_fingerprint() {
        let mut entry = FileIndexEntry::new("/test/path.md".to_string());
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        entry.add_fingerprint(id1);
        entry.add_fingerprint(id2);
        assert_eq!(entry.fingerprint_count(), 2);

        assert!(entry.remove_fingerprint(id1));
        assert_eq!(entry.fingerprint_count(), 1);
        assert!(!entry.fingerprint_ids.contains(&id1));
        assert!(entry.fingerprint_ids.contains(&id2));

        // Removing non-existent ID should return false
        let id3 = Uuid::new_v4();
        assert!(!entry.remove_fingerprint(id3));
    }

    #[test]
    fn test_file_index_entry_serialization() {
        let mut entry = FileIndexEntry::new("/test/path.md".to_string());
        entry.add_fingerprint(Uuid::new_v4());
        entry.add_fingerprint(Uuid::new_v4());

        let json = serde_json::to_string(&entry).expect("Serialization should succeed");
        let deserialized: FileIndexEntry =
            serde_json::from_str(&json).expect("Deserialization should succeed");

        assert_eq!(deserialized.file_path, entry.file_path);
        assert_eq!(deserialized.fingerprint_count(), entry.fingerprint_count());
    }

    #[test]
    fn test_file_watcher_stats_default() {
        let stats = FileWatcherStats::default();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.total_chunks, 0);
    }
}
