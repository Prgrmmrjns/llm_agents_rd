def get_schema(schema_name: str):
    """Get the appropriate schema based on schema name and API type."""
    schema_map = {
        "answer": get_answer_schema(),
        "research": get_research_schema(),
        "search": get_search_schema(),
        "reformulated_options": get_reformulated_options_schema()
    }
    
    return schema_map.get(schema_name)

def get_search_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "search",
            "schema": {
                "type": "object",
                "properties": {
                    "keywords_by_option": {
                        "type": "object",
                        "description": "Keywords that could help prove or disprove each option",
                        "properties": {
                            "A": {"type": "array", "items": {"type": "string"}},
                            "B": {"type": "array", "items": {"type": "string"}},
                            "C": {"type": "array", "items": {"type": "string"}},
                            "D": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["A", "B", "C", "D"]
                    },
                    "general_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "General keywords relevant to the overall question"
                    },
                    "temporal_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Time-related keywords (e.g., early onset, late stage)"
                    },
                    "demographic_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Age, gender, or population-related keywords"
                    }
                },
                "required": ["keywords_by_option", "general_keywords"]
            }
        }
    }

def get_answer_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {
                    "chosen_answer": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "The letter of the chosen answer (A/B/C/D)"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Detailed explanation of why this answer was chosen or why more information is needed"
                    }
                },
                "required": ["chosen_answer", "explanation"]
            }
        }
    }

def get_research_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "research",
            "schema": {
                "type": "object",
                "properties": {     
                    "evidence_by_option": {
                        "type": "object",
                        "description": "Evidence supporting each answer option",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"}
                        },
                        "required": ["A", "B", "C", "D"]
                    },
                    "counter_evidence_by_option": {
                        "type": "object",
                        "description": "Evidence contradicting each answer option",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"}
                        },
                        "required": ["A", "B", "C", "D"]
                    },
                    "accumulated_evidence": {
                        "type": "string",
                        "description": "Accumulated evidence from previous chunks that remains relevant"
                    }
                },
                "required": ["evidence_by_option", "counter_evidence_by_option", "accumulated_evidence"]
            }
        }
    }
    
def get_reformulated_options_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "reformulated_options",
            "schema": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string",
                        "description": "Reformulated option A for the question"
                    },
                    "b": {
                        "type": "string",
                        "description": "Reformulated option B for the question"
                    },
                    "c": {
                        "type": "string",
                        "description": "Reformulated option C for the question"
                    },
                    "d": {
                        "type": "string",
                        "description": "Reformulated option D for the question"
                    }
                },
                "required": ["a", "b", "c", "d"]
            }
        }
    }

