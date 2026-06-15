//! Temporal tool definitions for E2 V_freshness recency search and E3 V_periodicity.
//!
//! Per Constitution v6.5: E2 (V_freshness) finds recency patterns, E3 (V_periodicity)
//! finds time-of-day and day-of-week patterns.
//! Temporal embedders are POST-RETRIEVAL only per ARCH-25.
//!
//! Tools:
//! - search_recent: Search with E2 temporal boost applied
//! - search_periodic: Search with E3 periodic pattern boost applied

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns temporal tool definitions.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // search_recent - temporal search with E2 freshness boost
        ToolDefinition::new(
            "search_recent",
            "Search for recent memories with E2 temporal boost applied. \
             Automatically applies freshness decay to prioritize recent results. \
             Use for queries like 'what did we discuss recently', 'latest updates', \
             'yesterday's conversation', or any time-sensitive retrieval. \
             Per ARCH-25: Temporal boost is applied POST-retrieval, not in similarity fusion.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "topK": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum number of results to return"
                    },
                    "temporalWeight": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.3,
                        "description": "Temporal boost weight [0.1, 1.0]. Higher = more recency preference. Default: 0.3"
                    },
                    "decayFunction": {
                        "type": "string",
                        "enum": ["linear", "exponential", "step", "none", "no_decay"],
                        "default": "exponential",
                        "description": "Decay function for freshness. 'exponential' (default) = natural forgetting curve, 'linear' = simple decay, 'step' = time bucket based, 'none'/'no_decay' = no decay applied"
                    },
                    "temporalScale": {
                        "type": "string",
                        "enum": ["micro", "meso", "macro", "long", "archival"],
                        "default": "meso",
                        "description": "Time scale for decay. 'micro' = 1 hour horizon, 'meso' = 1 day (default), 'macro' = 1 week, 'long' = 1 month, 'archival' = 1 year"
                    },
                    "includeContent": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include content text in results (default: true)"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.1,
                        "description": "Minimum semantic similarity threshold before temporal boost"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        ),
        // search_periodic - temporal search with E3 periodic pattern boost
        ToolDefinition::new(
            "search_periodic",
            "Search for memories matching periodic time patterns (E3 V_periodicity). \
             Finds memories from similar times of day or days of week. \
             Use for queries like 'morning meetings', 'Friday deployments', \
             'what do I usually work on at 3pm?', or finding routine patterns. \
             Per ARCH-25: Periodic boost is applied POST-retrieval, not in similarity fusion.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "topK": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum number of results to return"
                    },
                    "targetHour": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 23,
                        "description": "Target hour (0-23). If omitted and autoDetect=true, uses current hour."
                    },
                    "targetDayOfWeek": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 6,
                        "description": "Target day (0=Sunday, 6=Saturday). If omitted and autoDetect=true, uses current day."
                    },
                    "autoDetect": {
                        "type": "boolean",
                        "default": false,
                        "description": "Auto-detect target from current time. When true, targetHour/targetDayOfWeek are computed from now."
                    },
                    "periodicWeight": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.3,
                        "description": "Weight for periodic boost [0.1, 1.0]. Higher = more periodic preference. Default: 0.3"
                    },
                    "includeContent": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include content text in results (default: true)"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.1,
                        "description": "Minimum semantic similarity threshold before periodic boost"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        ),
    ]
}
