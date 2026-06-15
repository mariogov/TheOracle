//! Curation tool definitions per PRD v6 Section 10.3.
//!
//! Tools:
//! - forget_concept: Soft-delete a memory (30-day recovery per SEC-06)
//! - boost_importance: Adjust memory importance score
//!
//! Constitution Compliance:
//! - SEC-06: Soft delete 30-day recovery
//! - BR-MCP-001: forget_concept uses soft delete by default
//! - BR-MCP-002: boost_importance clamps final value to [0.0, 1.0]
//! - AP-10: No NaN/Infinity in values

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns curation tool definitions (2 tools per PRD).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // forget_concept
        ToolDefinition::new(
            "forget_concept",
            "Soft-delete a memory with 30-day recovery window (per SEC-06). \
             Set soft_delete=false for permanent deletion (use with caution). \
             Returns deleted_at timestamp for recovery tracking.",
            json!({
                "type": "object",
                "required": ["node_id"],
                "properties": {
                    "node_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the memory to forget"
                    },
                    "soft_delete": {
                        "type": "boolean",
                        "default": true,
                        "description": "Use soft delete with 30-day recovery (default true per BR-MCP-001)"
                    },
                    "operator_id": {
                        "type": "string",
                        "description": "Operator ID for provenance tracking (who performed the deletion)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for deletion (for audit trail)"
                    }
                },
                "additionalProperties": false
            }),
        ),
        // boost_importance
        ToolDefinition::new(
            "boost_importance",
            "Adjust a memory's importance score by delta. Final value is clamped \
             to [0.0, 1.0] (per BR-MCP-002). Response includes old, delta, and new values.",
            json!({
                "type": "object",
                "required": ["node_id", "delta"],
                "properties": {
                    "node_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the memory to boost"
                    },
                    "delta": {
                        "type": "number",
                        "minimum": -1.0,
                        "maximum": 1.0,
                        "description": "Importance change value (-1.0 to 1.0)"
                    },
                    "operator_id": {
                        "type": "string",
                        "description": "Operator ID for provenance tracking (who performed the boost)"
                    }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definitions_exist_with_required_fields() {
        let tools = definitions();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"forget_concept"));
        assert!(names.contains(&"boost_importance"));
        // forget_concept: SEC-06 soft delete default true
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        assert!(forget.description.contains("SEC-06"));
        let props = forget.input_schema.get("properties").unwrap();
        assert!(props["soft_delete"]["default"].as_bool().unwrap());
        // boost_importance: delta [-1,1], BR-MCP-002
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        assert!(boost.description.contains("BR-MCP-002"));
        let delta = boost.input_schema["properties"]["delta"].clone();
        assert!((delta["minimum"].as_f64().unwrap() - (-1.0)).abs() < f64::EPSILON);
        assert!((delta["maximum"].as_f64().unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_synthetic_valid_input() {
        let tools = definitions();
        // forget_concept with UUID
        let node_id = "550e8400-e29b-41d4-a716-446655440000";
        assert!(uuid::Uuid::parse_str(node_id).is_ok());
        // boost_importance delta in range
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let delta_schema = &boost.input_schema["properties"]["delta"];
        let min = delta_schema["minimum"].as_f64().unwrap();
        let max = delta_schema["maximum"].as_f64().unwrap();
        for delta in [0.3, -0.2, 1.0, -1.0, 0.0] {
            assert!(delta >= min && delta <= max);
        }
    }
}
