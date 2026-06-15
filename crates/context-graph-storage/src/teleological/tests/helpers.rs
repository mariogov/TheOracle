//! Helper functions for creating REAL test data (no mocks).

use context_graph_core::types::fingerprint::{SemanticFingerprint, TeleologicalFingerprint};

/// Create a SemanticFingerprint with non-zero stub embeddings for testing.
///
/// STOR-M1: Uses 0.1-filled vectors instead of zeros because zero-norm vectors
/// are now correctly rejected by validate_vector() (cosine similarity is undefined
/// for zero-norm vectors). The previous `zeroed()` call produced all-zero vectors
/// that passed validation but gave meaningless HNSW results.
pub fn create_real_semantic() -> SemanticFingerprint {
    SemanticFingerprint::stub()
}

/// Create a REAL content hash.
pub fn create_real_hash() -> [u8; 32] {
    let mut hash = [0u8; 32];
    // SHA-256 of "test content"
    hash[0] = 0xDE;
    hash[1] = 0xAD;
    hash[30] = 0xBE;
    hash[31] = 0xEF;
    hash
}

/// Create a REAL TeleologicalFingerprint for testing.
pub fn create_real_fingerprint() -> TeleologicalFingerprint {
    TeleologicalFingerprint::new(create_real_semantic(), create_real_hash())
}

/// Create a TeleologicalFingerprint with the correct content hash for the given content.
/// This is needed because store_content() validates the hash.
pub fn create_fingerprint_for_content(content: &str) -> TeleologicalFingerprint {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let content_hash: [u8; 32] = hasher.finalize().into();

    TeleologicalFingerprint::new(create_real_semantic(), content_hash)
}
