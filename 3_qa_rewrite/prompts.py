"""LLM prompt templates per style + JSON schema for step 3 Q&A rewrite.

Pattern: one dedicated style prompt per call (no multi-style batching).
Inspired by `INSTRUCTION_STYLES` reference from project guidelines.
"""
from __future__ import annotations

PROMPT_VERSION = "v2"


SYSTEM_PROMPT = (
    "You are a clinical triage decision-support assistant for low-resource "
    "settings (e.g. sub-Saharan Africa primary care). You are NOT a substitute "
    "for clinician judgment.\n\n"
    "BEGIN every answer with exactly one Action line, then the reasoning:\n"
    "- clinical cases: 'Action: REFER NOW' | 'Action: URGENT SAME-DAY CARE' | "
    "'Action: ROUTINE FOLLOW-UP' | 'Action: HOME CARE + RETURN ADVICE'\n"
    "- non-clinical or out-of-domain requests: 'Action: OUT-OF-SCOPE'\n\n"
    "Pick the triage framework: a child under 5 in a low-resource setting uses "
    "WHO IMCI/ETAT (IMCI/ETAT takes priority over ESI); an adult "
    "emergency-department case may use ESI; out-of-domain requests use no "
    "framework. Do NOT force an ESI or SATS code when the framework is "
    "IMCI/ETAT or out-of-domain.\n\n"
    "IMCI/ETAT danger signs: if ANY is present OR strongly suspected you MUST "
    "lead with 'Action: REFER NOW' and recommend immediate referral to the "
    "nearest health facility. Treat as danger signs even if not named "
    "explicitly: convulsions, a seizure or fit, shaking or jerking, a stiff "
    "body, eyes rolled back, post-ictal lethargy, unable to drink or "
    "breastfeed, vomiting everything, lethargic or unconscious, stridor in a "
    "calm child, central cyanosis, shock, severe dehydration. Do not delay "
    "referral because some details are missing.\n\n"
    "In low-resource referral triage, diagnostic resources (fluids, lab tests, "
    "imaging) must NEVER replace or precede the referral decision: mention "
    "likely facility-level care only after the Action line and only as "
    "something the facility may do. If uncertain, escalate."
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-style instruction prompts (system role for question-generation LLM)
# ─────────────────────────────────────────────────────────────────────────────

STYLE_PROMPTS: dict[str, str] = {
    # ── Cases (MIETIC) ────────────────────────────────────────────────────
    "caregiver_query": (
        "You are creating Q&A training pairs for an emergency triage chatbot.\n"
        "Given a clinical case description, write ONE realistic question a "
        "family member or caregiver (non-clinical) would ask about this "
        "patient's situation.\n"
        "- First person perspective (\"my father\", \"my child\", \"my wife\").\n"
        "- Lay vocabulary, no medical jargon.\n"
        "- Express concern, ask about urgency or what to do.\n"
        "- 1-2 sentences max.\n"
        "- Mention the patient's key symptoms in plain words.\n"
        "Output ONLY the question text, nothing else."
    ),

    "clinician_handoff": (
        "You are creating Q&A training pairs for an emergency triage chatbot.\n"
        "Given a clinical case description, write ONE realistic handoff-style "
        "query a triage nurse or ED clinician would send to a senior "
        "colleague or decision-support tool.\n"
        "- Compact clinical shorthand (e.g. \"69M lung CA c/o liver mets, "
        "RR 40, SpO2 94%\").\n"
        "- Include key vitals if present in the case.\n"
        "- Ask for triage assessment, priority level, or next step.\n"
        "- 1-3 sentences.\n"
        "Output ONLY the question text, nothing else."
    ),

    "field_worker_query": (
        "You are creating Q&A training pairs for an emergency triage chatbot.\n"
        "Given a clinical case description, write ONE realistic question a "
        "rural community health worker or LMIC clinic nurse would ask, "
        "given resource constraints (limited diagnostics, possibly no "
        "physician on site).\n"
        "- Reference the resource-constrained setting.\n"
        "- Ask about: managing locally vs urgent referral, what to do now, "
        "or how to stabilize.\n"
        "- Plain English, practical tone.\n"
        "- 1-2 sentences.\n"
        "Output ONLY the question text, nothing else."
    ),

    # ── Sections (WHO/SATS guidelines PDF) ────────────────────────────────
    "direct_question": (
        "You are creating Q&A training pairs from medical guideline "
        "sections.\n"
        "Given a guideline section text, write ONE focused question that "
        "this section directly answers.\n"
        "- Single, specific clinical question (e.g. \"What are the signs of "
        "severe dehydration in children?\").\n"
        "- 1 sentence.\n"
        "- If the section is a Table of Contents, acknowledgements, preface, "
        "references list, page footer, or otherwise non-clinical filler, "
        "set skip=true.\n"
        "Output ONLY the question text (and set skip appropriately)."
    ),

    "protocol_lookup": (
        "You are creating Q&A training pairs from medical guideline "
        "sections.\n"
        "Given a guideline section text, write ONE realistic question a "
        "clinician would ask when looking up a protocol.\n"
        "- Reference the source authority where natural (\"What does WHO "
        "recommend...\", \"Per SATS protocol...\").\n"
        "- Single clinical/practical query.\n"
        "- 1 sentence.\n"
        "- If non-clinical filler, set skip=true.\n"
        "Output ONLY the question text."
    ),

    "red_flag_check": (
        "You are creating Q&A training pairs from medical guideline "
        "sections.\n"
        "Given a guideline section text, write ONE escalation-criteria "
        "question (\"when to refer\", \"warning signs\", \"red flags\", "
        "\"hospitalization indications\").\n"
        "- Focus on when to escalate care, refer to hospital, or worry.\n"
        "- 1 sentence.\n"
        "- If section is non-clinical filler, set skip=true.\n"
        "- If section content doesn't naturally support an escalation "
        "question (e.g. it's about prevention only), still produce a "
        "best-fit warning-signs question.\n"
        "Output ONLY the question text."
    ),
}


# Style → doc_type mapping (which styles apply where)
STYLES_FOR_CASES = ("caregiver_query", "clinician_handoff", "field_worker_query")
STYLES_FOR_SECTIONS = ("direct_question", "protocol_lookup", "red_flag_check")


# ─────────────────────────────────────────────────────────────────────────────
# Guided JSON schema (minimal — anti-fluff)
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The generated question text (1-3 sentences).",
        },
        "skip": {
            "type": "boolean",
            "description": "True if the source paragraph is non-clinical filler (TOC, refs, etc.).",
        },
    },
    "required": ["question", "skip"],
    "additionalProperties": False,
}


def render_prompt(text: str, style_name: str) -> str:
    """Return a single user-role prompt for the question generator.

    The style-specific instructions go into the user message (the model's
    system prompt is unrelated to the in-domain SYSTEM_PROMPT, which is
    only used in the FINAL training records).
    """
    instructions = STYLE_PROMPTS[style_name]
    return (
        f"{instructions}\n\n"
        f"SOURCE PARAGRAPH:\n{text.strip()}\n\n"
        "Return JSON: {\"question\": \"...\", \"skip\": false}"
    )
