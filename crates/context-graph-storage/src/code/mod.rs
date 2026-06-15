//! Code entity storage module.
//!
//! Provides storage for code entities and their E7 embeddings, separate from
//! the regular text content storage. Code uses AST-based chunking and E7
//! (Qodo-Embed-1-1.5B) as the primary embedder.
//!
//! # Architecture
//! - Separate column families from teleological storage
//! - E7 embeddings are the primary index
//! - Secondary indexes for name and signature search
//! - File-level tracking for efficient updates
//!
//! # Column Families
//! - `code_entities`: CodeEntity storage
//! - `code_e7_embeddings`: E7 1536D embeddings
//! - `code_file_index`: File path → entity IDs
//! - `code_name_index`: Entity name → entity IDs
//! - `code_signature_index`: Signature hash → entity IDs

mod error;
mod store;

pub use error::{CodeStorageError, CodeStorageResult};
pub use store::{CodeStore, E7_CODE_DIM};
