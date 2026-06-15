//! Serialization helpers for quantized embeddings.
//!
//! Provides bincode serialization/deserialization and key generation
//! for quantized fingerprint storage operations.

use context_graph_embeddings::QuantizedEmbedding;
use uuid::Uuid;

use super::error::{QuantizedStorageError, QuantizedStorageResult};

// =============================================================================
// SERIALIZATION HELPERS
// =============================================================================

/// Serialize a QuantizedEmbedding to bytes using bincode.
///
/// # FAIL FAST
/// Returns error with full context on failure. Caller should typically panic.
pub fn serialize_quantized_embedding(
    fingerprint_id: Uuid,
    embedder_idx: u8,
    embedding: &QuantizedEmbedding,
) -> QuantizedStorageResult<Vec<u8>> {
    bincode::serialize(embedding).map_err(|e| QuantizedStorageError::SerializationFailed {
        fingerprint_id,
        embedder_idx,
        reason: e.to_string(),
    })
}

/// Deserialize a QuantizedEmbedding from bytes using bincode.
///
/// # FAIL FAST
/// Returns error with full context on failure. Caller should typically panic.
pub fn deserialize_quantized_embedding(
    fingerprint_id: Uuid,
    embedder_idx: u8,
    data: &[u8],
) -> QuantizedStorageResult<QuantizedEmbedding> {
    bincode::deserialize(data).map_err(|e| QuantizedStorageError::DeserializationFailed {
        fingerprint_id,
        embedder_idx,
        reason: e.to_string(),
    })
}

/// Create the key for a fingerprint/embedder combination.
///
/// Key format: 16-byte UUID (big-endian bytes).
#[inline]
pub fn embedder_key(fingerprint_id: Uuid) -> [u8; 16] {
    *fingerprint_id.as_bytes()
}
