//! Graph edge connecting two memory nodes.
//!
//! This module provides the GraphEdge struct which represents directed relationships
//! between MemoryNodes in the Context Graph. It supports amortized shortcuts and
//! steering rewards for reinforcement learning.
//!
//! # Module Structure
//! - `edge`: Core GraphEdge struct, EdgeType enum, and constructors
//! - `modulation`: Steering modulation methods
//! - `traversal`: Traversal tracking and shortcut detection methods

mod edge;
mod modulation;
mod traversal;

#[cfg(test)]
mod tests_constructor;
#[cfg(test)]
mod tests_modulation;
#[cfg(test)]
mod tests_struct;
#[cfg(test)]
mod tests_traversal;

// Re-export all public items
pub use self::edge::{EdgeId, EdgeType, GraphEdge};
