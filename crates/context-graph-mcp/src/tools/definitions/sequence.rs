//! Sequence tool definitions for E4 (V_ordering) integration.
//!
//! These tools provide first-class conversational context capabilities:
//! - `get_conversation_context`: Get memories around current turn
//! - `get_session_timeline`: Ordered session memory timeline
//! - `traverse_memory_chain`: Multi-hop memory navigation
//! - `compare_session_states`: Before/after state comparison
//!
//! # Research Basis
//!
//! - MemoriesDB (2025): Cross-temporal coherence
//! - Memoria Framework: Session-level summarization + recency weighting
//! - TG-RAG: Bi-level temporal graph
//! - Episodic Memory for RAG: Space-time-anchored narratives

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns sequence tool definitions (4 tools per Phase 1 plan).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // get_conversation_context - auto-anchored context retrieval
        ToolDefinition::new(
            "get_conversation_context",
            "Get memories around the current conversation turn with auto-anchoring. \
             Uses E4 (V_ordering) for sequence-based retrieval. \
             Perfect for \"What did we discuss before X?\" queries.",
            json!({
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["before", "after", "both"],
                        "default": "before",
                        "description": "Direction to search: before (earlier turns), after (later turns), both"
                    },
                    "windowSize": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Number of turns to include in the window"
                    },
                    "sessionOnly": {
                        "type": "boolean",
                        "default": true,
                        "description": "Only include memories from current session"
                    },
                    "includeContent": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include full content text in results"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional semantic filter to apply alongside sequence ordering"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.0,
                        "description": "Minimum similarity threshold when query is provided"
                    }
                },
                "required": [],
                "additionalProperties": false
            }),
        ),
        // get_session_timeline - ordered timeline view
        ToolDefinition::new(
            "get_session_timeline",
            "Get an ordered timeline of all session memories with sequence numbers. \
             Returns memories sorted by session_sequence in ascending order. \
             Includes position labels like \"2 turns ago\", \"previous turn\".",
            json!({
                "type": "object",
                "properties": {
                    "sessionId": {
                        "type": "string",
                        "description": "Session ID to get timeline for (default: current session)"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 50,
                        "description": "Maximum number of memories to return"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Number of memories to skip for pagination"
                    },
                    "sourceTypes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["HookDescription", "ClaudeResponse", "Manual", "MDFileChunk"]
                        },
                        "description": "Filter by source types (default: all types)"
                    },
                    "includeContent": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include full content text in results"
                    }
                },
                "required": [],
                "additionalProperties": false
            }),
        ),
        // traverse_memory_chain - multi-hop navigation
        ToolDefinition::new(
            "traverse_memory_chain",
            "Navigate through a chain of memories starting from an anchor point. \
             Supports multi-hop traversal for understanding conversation flow. \
             Useful for tracing the evolution of a topic across turns.",
            json!({
                "type": "object",
                "properties": {
                    "anchorId": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the starting memory (anchor point)"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "bidirectional"],
                        "default": "backward",
                        "description": "Direction to traverse: forward (later), backward (earlier), bidirectional"
                    },
                    "hops": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of hops to traverse"
                    },
                    "semanticFilter": {
                        "type": "string",
                        "description": "Optional topic filter to only traverse related memories"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                        "description": "Minimum semantic similarity for filtered traversal"
                    },
                    "includeContent": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include full content text in results"
                    }
                },
                "required": ["anchorId"],
                "additionalProperties": false
            }),
        ),
        // compare_session_states - before/after analysis
        ToolDefinition::new(
            "compare_session_states",
            "Compare memory state at different sequence points in a session. \
             Useful for understanding what changed between two points in the conversation. \
             Returns topics, memory counts, and key differences.",
            json!({
                "type": "object",
                "properties": {
                    "beforeSequence": {
                        "type": ["integer", "string"],
                        "description": "Starting sequence number or \"start\" for session beginning"
                    },
                    "afterSequence": {
                        "type": ["integer", "string"],
                        "description": "Ending sequence number or \"current\" for current position"
                    },
                    "sessionId": {
                        "type": "string",
                        "description": "Session ID to compare within (default: current session)"
                    }
                },
                "required": ["beforeSequence", "afterSequence"],
                "additionalProperties": false
            }),
        ),
    ]
}
