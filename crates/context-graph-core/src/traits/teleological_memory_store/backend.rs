//! Storage backend types for teleological memory stores.

use serde::{Deserialize, Serialize};

/// Storage backend types for teleological memory stores.
///
/// This enum identifies the underlying storage implementation,
/// enabling runtime introspection and backend-specific optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TeleologicalStorageBackend {
    /// In-memory storage (HashMap-based, no persistence).
    /// Used for testing and development.
    InMemory,

    /// RocksDB storage with 8 column families.
    /// Production storage with full persistence and indexing.
    RocksDb,

    /// TimescaleDB storage for time-series evolution data.
    /// Used for purpose evolution archival.
    TimescaleDb,

    /// Hybrid storage combining RocksDB + TimescaleDB.
    /// Full production deployment.
    Hybrid,
}

impl std::fmt::Display for TeleologicalStorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory => write!(f, "InMemory"),
            Self::RocksDb => write!(f, "RocksDB"),
            Self::TimescaleDb => write!(f, "TimescaleDB"),
            Self::Hybrid => write!(f, "Hybrid (RocksDB + TimescaleDB)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_backend_display() {
        assert_eq!(TeleologicalStorageBackend::InMemory.to_string(), "InMemory");
        assert_eq!(TeleologicalStorageBackend::RocksDb.to_string(), "RocksDB");
        assert_eq!(
            TeleologicalStorageBackend::TimescaleDb.to_string(),
            "TimescaleDB"
        );
        assert_eq!(
            TeleologicalStorageBackend::Hybrid.to_string(),
            "Hybrid (RocksDB + TimescaleDB)"
        );
    }
}
