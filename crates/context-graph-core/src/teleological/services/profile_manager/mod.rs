//! TASK-TELEO-015: ProfileManager Implementation
//!
//! Manages task-specific teleological profiles. Provides CRUD operations
//! for profiles, context-based profile matching, and usage tracking.
//!
//! # Core Responsibilities
//!
//! 1. Create, read, update, delete profiles
//! 2. Find best matching profile for a given context
//! 3. Track profile usage statistics
//! 4. Provide built-in profiles for common tasks
//!
//! # Built-in Profiles
//!
//! - **code_implementation**: Emphasizes E6 (Code) for programming tasks
//! - **research_analysis**: Emphasizes E1, E4, E7 for semantic/causal analysis
//! - **creative_writing**: Emphasizes E10, E11 for qualitative/abstract tasks
//!
//! # Module Structure
//!
//! - `types`: Configuration, result, and internal types
//! - `builtin`: Built-in profile factory functions
//! - `manager`: Core ProfileManager implementation
//! - `tests`: Comprehensive test suite

pub mod builtin;
mod manager;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::manager::ProfileManager;
pub use self::types::{ProfileManagerConfig, ProfileMatch, ProfileStats};
