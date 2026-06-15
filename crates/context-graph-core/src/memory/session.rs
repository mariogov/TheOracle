//! Session tracking types for memory capture.
//!
//! This module provides types for tracking memory capture sessions:
//! - [`Session`] - A memory capture session with lifecycle tracking
//! - [`SessionStatus`] - Session lifecycle states
//!
//! # Constitution Compliance
//! - ARCH-07: NATIVE Claude Code hooks control memory lifecycle
//! - Sessions group memories captured during a Claude Code session
//!
//! # Note
//! This module defines TYPES ONLY. SessionManager (persistence, lifecycle
//! management) is implemented in TASK-P1-006.
//!
//! # Example
//! ```rust
//! use context_graph_core::memory::{Session, SessionStatus};
//!
//! // Create a new active session
//! let mut session = Session::new();
//! assert!(session.is_active());
//! assert_eq!(session.memory_count, 0);
//!
//! // Track memories
//! session.increment_memory_count();
//! assert_eq!(session.memory_count, 1);
//!
//! // Complete the session
//! session.complete();
//! assert!(!session.is_active());
//! assert!(session.ended_at.is_some());
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Status of a memory capture session.
///
/// Sessions progress through these states:
/// - `Active` → Session in progress, memories being captured
/// - `Completed` → Session ended normally via SessionEnd hook
/// - `Abandoned` → Session ended without proper closure (crash, timeout)
///
/// # State Transitions
/// ```text
/// [new] → Active → Completed
///              ↘→ Abandoned
/// ```
///
/// Once a session is Completed or Abandoned, it cannot return to Active.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum SessionStatus {
    /// Session is currently active and capturing memories.
    #[default]
    Active,
    /// Session ended normally via SessionEnd hook.
    Completed,
    /// Session ended without proper closure (crash, timeout, etc.).
    Abandoned,
}

impl SessionStatus {
    /// Check if this status represents an active session.
    pub fn is_active(&self) -> bool {
        matches!(self, SessionStatus::Active)
    }

    /// Check if this status represents a terminated session.
    pub fn is_terminated(&self) -> bool {
        matches!(self, SessionStatus::Completed | SessionStatus::Abandoned)
    }
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionStatus::Active => write!(f, "Active"),
            SessionStatus::Completed => write!(f, "Completed"),
            SessionStatus::Abandoned => write!(f, "Abandoned"),
        }
    }
}

/// A memory capture session.
///
/// Sessions track the lifecycle of memory capture during a Claude Code session.
/// Each session has:
/// - Unique identifier (UUID string)
/// - Start/end timestamps
/// - Status (Active, Completed, Abandoned)
/// - Count of memories captured
///
/// # Lifecycle
/// 1. SessionStart hook → `Session::new()` creates Active session
/// 2. Memory captured → `increment_memory_count()` called
/// 3. SessionEnd hook → `complete()` marks session Completed
/// 4. (or crash/timeout) → `abandon()` marks session Abandoned
///
/// # Storage
/// Sessions are persisted to RocksDB in the `sessions` column family.
/// Key: session_id bytes, Value: bincode-serialized Session.
///
/// # Note
/// `id` is a String (not Uuid) for simpler serialization and CLI usage.
/// Internally it stores UUID v4 string representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session identifier (UUID v4 string format).
    /// Primary key in storage.
    pub id: String,

    /// When the session started.
    pub started_at: DateTime<Utc>,

    /// When the session ended (None if still Active).
    pub ended_at: Option<DateTime<Utc>>,

    /// Current session status.
    pub status: SessionStatus,

    /// Number of memories captured in this session.
    pub memory_count: u32,
}

impl Session {
    /// Create a new active session with generated UUID.
    ///
    /// # Returns
    /// A new Session with:
    /// - `id`: Fresh UUID v4 string
    /// - `started_at`: Current UTC timestamp
    /// - `ended_at`: None
    /// - `status`: Active
    /// - `memory_count`: 0
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::memory::Session;
    ///
    /// let session = Session::new();
    /// assert!(session.is_active());
    /// assert!(session.ended_at.is_none());
    /// ```
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            started_at: Utc::now(),
            ended_at: None,
            status: SessionStatus::Active,
            memory_count: 0,
        }
    }

    /// Create a session with a specific ID (for restoration from storage).
    ///
    /// # Arguments
    /// * `id` - The session ID (should be valid UUID string)
    ///
    /// # Note
    /// This creates a NEW session with the given ID, started_at = now.
    /// For full restoration, use `restore()` instead.
    pub fn with_id(id: String) -> Self {
        Self {
            id,
            started_at: Utc::now(),
            ended_at: None,
            status: SessionStatus::Active,
            memory_count: 0,
        }
    }

    /// Restore a session from storage data.
    ///
    /// # Arguments
    /// * `id` - Session identifier
    /// * `started_at` - Original start timestamp
    /// * `ended_at` - End timestamp if terminated
    /// * `status` - Current status
    /// * `memory_count` - Number of memories captured
    ///
    /// # Use Case
    /// Reconstructing Session from deserialized storage data.
    pub fn restore(
        id: String,
        started_at: DateTime<Utc>,
        ended_at: Option<DateTime<Utc>>,
        status: SessionStatus,
        memory_count: u32,
    ) -> Self {
        Self {
            id,
            started_at,
            ended_at,
            status,
            memory_count,
        }
    }

    /// Check if session is currently active.
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    /// Check if session has been terminated (completed or abandoned).
    pub fn is_terminated(&self) -> bool {
        self.status.is_terminated()
    }

    /// Mark session as completed (normal end via SessionEnd hook).
    ///
    /// Sets:
    /// - `ended_at` to current UTC timestamp
    /// - `status` to Completed
    ///
    /// # Idempotency
    /// If already terminated, this is a no-op.
    pub fn complete(&mut self) {
        if self.status == SessionStatus::Active {
            self.ended_at = Some(Utc::now());
            self.status = SessionStatus::Completed;
        }
    }

    /// Mark session as abandoned (abnormal end - crash, timeout, etc.).
    ///
    /// Sets:
    /// - `ended_at` to current UTC timestamp
    /// - `status` to Abandoned
    ///
    /// # Idempotency
    /// If already terminated, this is a no-op.
    pub fn abandon(&mut self) {
        if self.status == SessionStatus::Active {
            self.ended_at = Some(Utc::now());
            self.status = SessionStatus::Abandoned;
        }
    }

    /// Increment the memory count for this session.
    ///
    /// Called each time a Memory is captured belonging to this session.
    pub fn increment_memory_count(&mut self) {
        self.memory_count = self.memory_count.saturating_add(1);
    }

    /// Get the duration of the session (if ended).
    ///
    /// # Returns
    /// - `Some(duration)` if session has ended
    /// - `None` if session is still active
    pub fn duration(&self) -> Option<chrono::Duration> {
        self.ended_at.map(|end| end - self.started_at)
    }

    /// Get the duration from start until now (for active sessions).
    ///
    /// Useful for displaying "session running for X minutes" in CLI.
    pub fn elapsed(&self) -> chrono::Duration {
        Utc::now() - self.started_at
    }

    /// Validate the session for consistency.
    ///
    /// # Checks
    /// 1. ID is not empty
    /// 2. ID is valid UUID format
    /// 3. If terminated, ended_at must be Some
    /// 4. If active, ended_at must be None
    /// 5. ended_at >= started_at (if present)
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(String)` with description on failure.
    pub fn validate(&self) -> Result<(), String> {
        // 1. ID not empty
        if self.id.is_empty() {
            return Err("Session ID cannot be empty".to_string());
        }

        // 2. ID is valid UUID
        if Uuid::parse_str(&self.id).is_err() {
            return Err(format!("Session ID is not valid UUID: {}", self.id));
        }

        // 3 & 4. Status/ended_at consistency
        match self.status {
            SessionStatus::Active => {
                if self.ended_at.is_some() {
                    return Err("Active session should not have ended_at set".to_string());
                }
            }
            SessionStatus::Completed | SessionStatus::Abandoned => {
                if self.ended_at.is_none() {
                    return Err(format!("{} session must have ended_at set", self.status));
                }
            }
        }

        // 5. Time ordering
        if let Some(ended) = self.ended_at {
            if ended < self.started_at {
                return Err(format!(
                    "ended_at ({}) is before started_at ({})",
                    ended, self.started_at
                ));
            }
        }

        Ok(())
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Session {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Session {}

impl std::hash::Hash for Session {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new_generates_uuid() {
        let session = Session::new();

        // ID should be valid UUID
        assert!(!session.id.is_empty());
        assert!(
            Uuid::parse_str(&session.id).is_ok(),
            "ID should be valid UUID"
        );

        // Initial state
        assert!(session.is_active());
        assert!(!session.is_terminated());
        assert!(session.ended_at.is_none());
        assert_eq!(session.status, SessionStatus::Active);
        assert_eq!(session.memory_count, 0);
    }

    #[test]
    fn test_session_with_id() {
        let custom_id = "custom-session-id-123".to_string();
        let session = Session::with_id(custom_id.clone());

        assert_eq!(session.id, custom_id);
        assert!(session.is_active());
    }

    #[test]
    fn test_session_complete() {
        let mut session = Session::new();
        assert!(session.is_active());
        assert!(session.ended_at.is_none());

        session.complete();

        assert!(!session.is_active());
        assert!(session.is_terminated());
        assert!(session.ended_at.is_some());
        assert_eq!(session.status, SessionStatus::Completed);
    }

    #[test]
    fn test_session_abandon() {
        let mut session = Session::new();
        assert!(session.is_active());

        session.abandon();

        assert!(!session.is_active());
        assert!(session.is_terminated());
        assert!(session.ended_at.is_some());
        assert_eq!(session.status, SessionStatus::Abandoned);
    }

    #[test]
    fn test_session_complete_idempotent() {
        let mut session = Session::new();
        session.complete();
        let first_ended = session.ended_at;

        // Complete again - should be no-op
        session.complete();
        assert_eq!(session.ended_at, first_ended);
        assert_eq!(session.status, SessionStatus::Completed);
    }

    #[test]
    fn test_session_abandon_idempotent() {
        let mut session = Session::new();
        session.abandon();
        let first_ended = session.ended_at;

        // Abandon again - should be no-op
        session.abandon();
        assert_eq!(session.ended_at, first_ended);
        assert_eq!(session.status, SessionStatus::Abandoned);
    }

    #[test]
    fn test_session_cannot_complete_after_abandon() {
        let mut session = Session::new();
        session.abandon();

        // Try to complete - should be no-op (already terminated)
        session.complete();

        assert_eq!(session.status, SessionStatus::Abandoned);
    }

    #[test]
    fn test_session_increment_memory_count() {
        let mut session = Session::new();
        assert_eq!(session.memory_count, 0);

        session.increment_memory_count();
        assert_eq!(session.memory_count, 1);

        session.increment_memory_count();
        session.increment_memory_count();
        assert_eq!(session.memory_count, 3);
    }

    #[test]
    fn test_session_increment_memory_count_saturating() {
        let mut session = Session::new();
        session.memory_count = u32::MAX;

        // Should not overflow
        session.increment_memory_count();
        assert_eq!(session.memory_count, u32::MAX);
    }

    #[test]
    fn test_session_restore() {
        let id = Uuid::new_v4().to_string();
        let started = Utc::now() - chrono::Duration::hours(1);
        let ended = Some(Utc::now());

        let session = Session::restore(id.clone(), started, ended, SessionStatus::Completed, 42);

        assert_eq!(session.id, id);
        assert_eq!(session.started_at, started);
        assert_eq!(session.ended_at, ended);
        assert_eq!(session.status, SessionStatus::Completed);
        assert_eq!(session.memory_count, 42);
    }

    #[test]
    fn test_session_validate_success() {
        let session = Session::new();
        assert!(session.validate().is_ok());
    }

    #[test]
    fn test_session_validate_empty_id() {
        let session = Session::with_id(String::new());
        let result = session.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_session_validate_invalid_uuid() {
        let session = Session::with_id("not-a-uuid".to_string());
        let result = session.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not valid UUID"));
    }

    #[test]
    fn test_session_validate_active_with_ended_at() {
        let mut session = Session::new();
        // Manually set ended_at without changing status (invalid state)
        session.ended_at = Some(Utc::now());

        let result = session.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Active session"));
    }

    #[test]
    fn test_session_validate_completed_without_ended_at() {
        let session = Session::restore(
            Uuid::new_v4().to_string(),
            Utc::now(),
            None, // Missing ended_at for Completed status
            SessionStatus::Completed,
            0,
        );

        let result = session.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must have ended_at"));
    }

    #[test]
    fn test_session_validate_ended_before_started() {
        let now = Utc::now();
        let session = Session::restore(
            Uuid::new_v4().to_string(),
            now,
            Some(now - chrono::Duration::hours(1)), // ended before started
            SessionStatus::Completed,
            0,
        );

        let result = session.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("before started_at"));
    }

    #[test]
    fn test_session_duration() {
        let start = Utc::now() - chrono::Duration::minutes(30);
        let end = Utc::now();

        let session = Session::restore(
            Uuid::new_v4().to_string(),
            start,
            Some(end),
            SessionStatus::Completed,
            10,
        );

        let duration = session.duration().expect("should have duration");
        // Duration should be approximately 30 minutes
        assert!(duration.num_minutes() >= 29 && duration.num_minutes() <= 31);
    }

    #[test]
    fn test_session_duration_active() {
        let session = Session::new();
        assert!(session.duration().is_none());
    }

    #[test]
    fn test_session_elapsed() {
        let session = Session::new();
        let elapsed = session.elapsed();
        // Elapsed should be very small (just created)
        assert!(elapsed.num_seconds() < 2);
    }

    #[test]
    fn test_session_serialization_roundtrip() {
        let mut session = Session::new();
        session.increment_memory_count();
        session.increment_memory_count();
        session.complete();

        // Test bincode serialization (used for storage)
        let bytes = bincode::serialize(&session).expect("serialize failed");
        let restored: Session = bincode::deserialize(&bytes).expect("deserialize failed");

        assert_eq!(session.id, restored.id);
        assert_eq!(session.status, restored.status);
        assert_eq!(session.memory_count, restored.memory_count);
        assert!(restored.validate().is_ok());
    }

    #[test]
    fn test_session_json_serialization() {
        let session = Session::new();

        let json = serde_json::to_string(&session).expect("json serialize failed");
        let restored: Session = serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(session.id, restored.id);
        assert_eq!(session.status, restored.status);
    }

    #[test]
    fn test_session_status_display() {
        assert_eq!(SessionStatus::Active.to_string(), "Active");
        assert_eq!(SessionStatus::Completed.to_string(), "Completed");
        assert_eq!(SessionStatus::Abandoned.to_string(), "Abandoned");
    }

    #[test]
    fn test_session_status_default() {
        let status: SessionStatus = Default::default();
        assert_eq!(status, SessionStatus::Active);
    }

    #[test]
    fn test_session_equality_by_id() {
        let session1 = Session::with_id("same-id".to_string());
        let mut session2 = Session::with_id("same-id".to_string());
        session2.memory_count = 100; // Different memory_count

        // Sessions are equal by ID only
        assert_eq!(session1, session2);
    }

    #[test]
    fn test_session_hash() {
        use std::collections::HashSet;

        let session1 = Session::with_id("id-1".to_string());
        let session2 = Session::with_id("id-2".to_string());
        let session1_dup = Session::with_id("id-1".to_string());

        let mut set = HashSet::new();
        set.insert(session1.clone());
        set.insert(session2);

        // Duplicate ID should not increase set size
        set.insert(session1_dup);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_session_default() {
        let session: Session = Default::default();
        assert!(session.is_active());
        assert!(!session.id.is_empty());
    }
}
