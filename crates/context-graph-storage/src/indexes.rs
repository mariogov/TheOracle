//! Secondary index operations.
//!
//! Provides query capabilities beyond primary key lookup:
//! - Query by tag
//! - Query by time range
//! - Query by source
//!
//! # Index Strategy
//! Indexes are maintained as separate column families with
//! composite keys enabling efficient range scans.
