//! Test helper functions for TeleologicalFingerprint tests.

use crate::types::fingerprint::SemanticFingerprint;

pub fn make_test_semantic() -> SemanticFingerprint {
    // Explicitly using zeroed() - this is a test helper where we need placeholder data.
    // In production, use real embeddings from the embedding pipeline.
    SemanticFingerprint::zeroed()
}

pub fn make_test_hash() -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[0] = 0xDE;
    hash[1] = 0xAD;
    hash[30] = 0xBE;
    hash[31] = 0xEF;
    hash
}
