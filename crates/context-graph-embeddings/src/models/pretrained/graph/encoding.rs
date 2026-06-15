//! Graph encoding utilities for relation and context embedding.
//!
//! Provides methods to convert knowledge graph structures into text
//! suitable for embedding with the MiniLM model.

use super::constants::MAX_CONTEXT_NEIGHBORS;

/// Encode a relation triple into a text string for embedding.
///
/// Converts subject-predicate-object triples into natural language form
/// suitable for embedding with the MiniLM model.
///
/// # Arguments
/// * `subject` - The subject entity (e.g., "Alice")
/// * `predicate` - The relation predicate (e.g., "works_at")
/// * `object` - The object entity (e.g., "Anthropic")
///
/// # Returns
/// A string formatted as "{subject} {predicate} {object}" with underscores
/// in the predicate replaced by spaces.
///
/// # Examples
/// ```rust
/// use context_graph_embeddings::models::GraphModel;
///
/// let text = GraphModel::encode_relation("Alice", "works_at", "Anthropic");
/// assert_eq!(text, "Alice works at Anthropic");
///
/// let text = GraphModel::encode_relation("Bob", "is_friend_of", "Charlie");
/// assert_eq!(text, "Bob is friend of Charlie");
/// ```
pub fn encode_relation(subject: &str, predicate: &str, object: &str) -> String {
    let predicate_clean = predicate.replace('_', " ");
    format!("{} {} {}", subject, predicate_clean, object)
}

/// Encode a node with its neighboring relations into a context string.
///
/// Creates a text representation of a node and its immediate graph neighbors,
/// suitable for embedding with the MiniLM model.
///
/// # Arguments
/// * `node` - The central node entity name
/// * `neighbors` - Slice of (relation, neighbor_node) tuples
///
/// # Returns
/// A string formatted as "{node}: {rel1} {neighbor1}, {rel2} {neighbor2}, ..."
/// Limited to MAX_CONTEXT_NEIGHBORS (5) entries.
///
/// # Examples
/// ```rust
/// use context_graph_embeddings::models::GraphModel;
///
/// let context = GraphModel::encode_context(
///     "Alice",
///     &[
///         ("works_at".to_string(), "Anthropic".to_string()),
///         ("knows".to_string(), "Bob".to_string()),
///     ]
/// );
/// assert_eq!(context, "Alice: works at Anthropic, knows Bob");
/// ```
pub fn encode_context(node: &str, neighbors: &[(String, String)]) -> String {
    if neighbors.is_empty() {
        return node.to_string();
    }

    let limited = neighbors.iter().take(MAX_CONTEXT_NEIGHBORS);
    let neighbor_strs: Vec<String> = limited
        .map(|(rel, neighbor)| {
            let rel_clean = rel.replace('_', " ");
            format!("{} {}", rel_clean, neighbor)
        })
        .collect();

    format!("{}: {}", node, neighbor_strs.join(", "))
}
