//! Causal Hint Provider for E5 Embedding Enhancement.
//!
//! This module provides the [`CausalHintProvider`] trait and [`NoOpCausalHintProvider`]
//! for obtaining LLM-generated causal hints during memory storage.
//!
//! # Architecture
//!
//! ```text
//! store_memory(content)
//!     │
//!     ▼
//! CausalHintProvider.get_hint(content)  ◄── 100ms timeout
//!     │
//!     ├── LLM.analyze_single_text(content)
//!     │       │
//!     │       ▼
//!     │   CausalHint { is_causal, direction_hint, confidence, key_phrases }
//!     │
//!     ▼
//! EmbeddingMetadata { ..., causal_hint }
//!     │
//!     ▼
//! embed_all_with_metadata(content, metadata)
//!     │
//!     ▼
//! E5 uses hint for enhanced cause/effect vectors
//! ```
//!
//! # Graceful Degradation
//!
//! - If LLM is not loaded: returns None
//! - If timeout exceeded: returns None
//! - If confidence < 0.5: returns None
//! - If parse error: returns None
//!
//! In all failure cases, the E5 embedder falls back to marker-based detection.
//!
//! # Note
//!
//! The `LlmCausalHintProvider` implementation lives in `context-graph-mcp`
//! to avoid cyclic dependencies (causal-agent depends on embeddings for CausalModel).

use std::time::Duration;

use async_trait::async_trait;
use context_graph_core::traits::{CausalHint, ExtractedCausalRelationship};

/// Status of causal relationship extraction.
///
/// Used to track what happened during the last extraction attempt,
/// enabling better error reporting and debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtractionStatus {
    /// Extraction completed successfully with relationships found.
    Success,
    /// Content analyzed but no causal relationships found.
    NoContent,
    /// LLM is not loaded/available.
    Unavailable,
    /// Extraction timed out.
    Timeout,
    /// LLM inference error occurred.
    Error,
    /// Status not yet determined.
    #[default]
    Unknown,
}

impl ExtractionStatus {
    /// Convert status to a display string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "Success",
            Self::NoContent => "NoContent",
            Self::Unavailable => "Unavailable",
            Self::Timeout => "Timeout",
            Self::Error => "Error",
            Self::Unknown => "Unknown",
        }
    }
}

/// Provider for LLM-based causal hints and multi-relationship extraction.
///
/// Used during memory storage to obtain direction hints for E5 embedding
/// enhancement. Implementations should handle timeouts gracefully and
/// return `None` rather than blocking the storage pipeline.
///
/// # Methods
///
/// - [`get_hint`](Self::get_hint): Get a single causal hint (legacy, for E5 enhancement)
/// - [`extract_all_relationships`](Self::extract_all_relationships): Extract ALL cause-effect relationships
#[async_trait]
pub trait CausalHintProvider: Send + Sync {
    /// Get causal hint for content with built-in timeout.
    ///
    /// Returns `Some(CausalHint)` if:
    /// - LLM is available and loaded
    /// - Analysis completes within timeout
    /// - Confidence >= 0.5
    ///
    /// Returns `None` otherwise, allowing fallback to marker detection.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to analyze
    ///
    /// # Performance
    ///
    /// Target latency: <100ms to avoid blocking storage pipeline.
    async fn get_hint(&self, content: &str) -> Option<CausalHint>;

    /// Extract ALL causal relationships from content.
    ///
    /// Unlike [`get_hint`](Self::get_hint) which returns a single hint describing
    /// whether content IS causal, this method extracts every distinct cause-effect
    /// relationship found within the content.
    ///
    /// Each extracted relationship includes:
    /// - Brief cause and effect statements
    /// - A 1-2 paragraph explanation for E5 embedding
    /// - Confidence score and mechanism type
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to analyze
    ///
    /// # Returns
    ///
    /// A vector of [`ExtractedCausalRelationship`] instances. Returns an empty
    /// vector if:
    /// - LLM is not available
    /// - Content has no causal relationships
    /// - Extraction times out
    ///
    /// # Performance
    ///
    /// Target latency: <200ms (longer than get_hint due to multi-extraction).
    async fn extract_all_relationships(&self, content: &str) -> Vec<ExtractedCausalRelationship>;

    /// Check if the LLM is available for hint generation.
    ///
    /// Returns `true` if the model is loaded and ready for inference.
    fn is_available(&self) -> bool;

    /// Get the timeout duration for hint generation.
    fn timeout(&self) -> Duration;

    /// Get status of last extraction attempt.
    ///
    /// Returns the status of the most recent `extract_all_relationships` call.
    /// Useful for distinguishing between "no relationships found" vs "LLM error".
    ///
    /// # Default Implementation
    ///
    /// Returns `ExtractionStatus::Unknown` for backward compatibility.
    fn last_extraction_status(&self) -> ExtractionStatus {
        ExtractionStatus::Unknown
    }
}

/// No-op implementation for when LLM is not available.
///
/// Always returns `None`, causing E5 to use marker-based detection only.
pub struct NoOpCausalHintProvider;

impl NoOpCausalHintProvider {
    /// Create a new no-op provider.
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpCausalHintProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CausalHintProvider for NoOpCausalHintProvider {
    async fn get_hint(&self, _content: &str) -> Option<CausalHint> {
        None
    }

    async fn extract_all_relationships(&self, _content: &str) -> Vec<ExtractedCausalRelationship> {
        Vec::new()
    }

    fn is_available(&self) -> bool {
        false
    }

    fn timeout(&self) -> Duration {
        Duration::ZERO
    }

    fn last_extraction_status(&self) -> ExtractionStatus {
        ExtractionStatus::Unavailable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noop_provider_returns_none() {
        let provider = NoOpCausalHintProvider::new();
        let hint = provider.get_hint("test content").await;
        assert!(hint.is_none());
        assert!(!provider.is_available());
    }

    #[tokio::test]
    async fn test_noop_provider_zero_timeout() {
        let provider = NoOpCausalHintProvider::new();
        assert_eq!(provider.timeout(), Duration::ZERO);
    }
}
