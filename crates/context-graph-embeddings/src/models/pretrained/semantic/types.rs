//! Type definitions for the semantic embedding model.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use crate::gpu::BertWeights;
use crate::traits::SingleModelConfig;

pub(crate) use crate::models::pretrained::shared::ModelState;

/// Concrete state type for semantic model (BERT weights).
pub(crate) type SemanticModelState = ModelState<Box<BertWeights>>;

/// Semantic embedding model using intfloat/e5-large-v2.
///
/// This is the primary semantic understanding model producing 1024D dense vectors.
/// Uses instruction prefixes to distinguish between queries and passages.
///
/// # Thread Safety
/// - `AtomicBool` for `loaded` state (lock-free reads)
/// - Inner model/tokenizer require explicit synchronization if mutable
///
/// # Memory Layout
/// - Total estimated: 1.3GB for FP32 weights
/// - With FP16 quantization: ~650MB
pub struct SemanticModel {
    /// Model weights and inference engine.
    /// NOTE: Type depends on candle feature flag.
    /// For now, placeholder that compiles without candle.
    #[allow(dead_code)]
    pub(crate) model_state: std::sync::RwLock<SemanticModelState>,

    /// Path to model weights directory.
    #[allow(dead_code)]
    pub(crate) model_path: PathBuf,

    /// Configuration for this model instance.
    #[allow(dead_code)]
    pub(crate) config: SingleModelConfig,

    /// Whether model weights are loaded and ready.
    pub(crate) loaded: AtomicBool,
}

// SAFETY: SemanticModel wraps candle Tensors (via BertWeights in ModelState) behind a
// std::sync::RwLock. Candle Tensors contain raw GPU pointers that are !Send/!Sync,
// but all access is synchronized through the RwLock. Pretrained models with GPU weights
// still require unsafe Send/Sync impls.
unsafe impl Send for SemanticModel {}
unsafe impl Sync for SemanticModel {}
