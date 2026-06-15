//! Configuration types for single embedder search.

/// Single embedder search configuration.
///
/// # Fields
///
/// - `default_k`: Default number of results when not specified
/// - `default_threshold`: Default minimum similarity threshold
/// - `ef_search`: HNSW ef_search parameter override
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::search::SingleEmbedderSearchConfig;
///
/// let config = SingleEmbedderSearchConfig {
///     default_k: 100,
///     default_threshold: Some(0.5),
///     ef_search: Some(256),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SingleEmbedderSearchConfig {
    /// Default number of results to return.
    pub default_k: usize,

    /// Default minimum similarity threshold.
    pub default_threshold: Option<f32>,

    /// Override HNSW ef_search parameter.
    ///
    /// Higher values = more accurate but slower.
    /// None = use index default.
    pub ef_search: Option<usize>,
}

impl Default for SingleEmbedderSearchConfig {
    fn default() -> Self {
        Self {
            default_k: 100,
            default_threshold: None,
            ef_search: None,
        }
    }
}
