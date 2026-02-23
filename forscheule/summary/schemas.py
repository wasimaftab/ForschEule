"""JSON schemas for OpenAI Structured Outputs."""

SCHEMA_VERSION = "1"

PER_PAPER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "one_line_takeaway": {"type": "string"},
                "methods": {"type": "string"},
                "main_findings": {"type": "string"},
                "limitations": {"type": "string"},
                "relevance_to_lab": {"type": "string"},
                "novelty_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "read_priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": [
                "paper_id",
                "one_line_takeaway",
                "methods",
                "main_findings",
                "limitations",
                "relevance_to_lab",
                "novelty_level",
                "read_priority",
            ],
            "additionalProperties": False,
        },
    },
}

WEEKLY_DIGEST_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "weekly_digest",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "themes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "contradictions_or_tensions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "what_to_read_first": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["paper_id", "reason"],
                        "additionalProperties": False,
                    },
                },
                "methods_trends": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "recommended_next_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "themes",
                "contradictions_or_tensions",
                "what_to_read_first",
                "methods_trends",
                "recommended_next_queries",
            ],
            "additionalProperties": False,
        },
    },
}
