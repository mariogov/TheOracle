//! Teleological profiles for task-specific embedding fusion.
//!
//! From teleoplan.md Section 6.3 Meta-Learning Across Tasks:
//!
//! Profiles define task-specific configurations for:
//! - Embedding weights (which embeddings matter most)
//! - Fusion strategy (how to combine embeddings)
//! - Task type classification
//!
//! Example profiles:
//! - code_implementation: boosts E6 (Code), E7 (Procedural)
//! - conceptual_research: boosts E11 (Abstract), E5 (Analogical)

mod fusion_strategy;
mod metrics;
mod task_type;
mod teleological;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use fusion_strategy::FusionStrategy;
pub use metrics::ProfileMetrics;
pub use task_type::TaskType;
pub use teleological::TeleologicalProfile;
