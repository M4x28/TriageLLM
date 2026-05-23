"""LLM prompt templates + JSON schemas for step 2 extraction.

Versioned: bump PROMPT_VERSION whenever the wording changes so downstream
records can be invalidated and re-extracted.
"""
from __future__ import annotations

PROMPT_VERSION = "v1"


MIETIC_EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "chief_complaint": {
            "type": "string",
            "description": "One-sentence summary of the primary reason for ED visit.",
        },
        "symptoms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Discrete symptoms reported, lowercase, deduplicated.",
        },
        "history": {
            "type": "string",
            "description": "Relevant past medical history. Empty string if none.",
        },
    },
    "required": ["chief_complaint", "symptoms", "history"],
    "additionalProperties": False,
}


MIETIC_EXTRACT_PROMPT = """\
You are a clinical NLP extractor. Read the emergency department case below
and extract the requested fields as JSON. Be concise and faithful to the
text. Do not invent information not present in the input.

CASE:
{narrative}

Return ONLY a JSON object matching this schema:
{{
  "chief_complaint": "<one short sentence>",
  "symptoms": ["<symptom 1>", "<symptom 2>", ...],
  "history": "<relevant PMH or empty string>"
}}
"""


def render_mietic_prompt(narrative: str) -> str:
    return MIETIC_EXTRACT_PROMPT.format(narrative=narrative.strip())
