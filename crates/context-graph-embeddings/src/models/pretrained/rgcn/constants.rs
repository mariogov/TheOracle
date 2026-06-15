//! Constants for the R-GCN model.

/// Number of relation types (GraphLinkEdgeType variants).
pub const NUM_RELATIONS: usize = 8;

/// Input feature dimension (reduced E1 embeddings).
pub const INPUT_DIM: usize = 32;

/// Hidden layer dimension.
pub const HIDDEN_DIM: usize = 64;

/// Output embedding dimension.
pub const OUTPUT_DIM: usize = 32;

/// Number of basis matrices for weight decomposition.
pub const NUM_BASES: usize = 4;

/// Default model path relative to models directory.
pub const DEFAULT_WEIGHTS_PATH: &str = "models/rgcn/model.safetensors";

/// Default config path relative to models directory.
pub const DEFAULT_CONFIG_PATH: &str = "models/rgcn/config.json";

/// Relation type names for debugging.
pub const RELATION_NAMES: [&str; NUM_RELATIONS] = [
    "SemanticSimilar",
    "CodeRelated",
    "EntityShared",
    "CausalChain",
    "GraphConnected",
    "ParaphraseAligned",
    "KeywordOverlap",
    "MultiAgreement",
];
