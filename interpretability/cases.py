"""Trace cases + phrase sets for the ESI-framework-leak interpretability study.

Single source for the interpretability phase. Reuses the deployed SYSTEM_PROMPT
(loaded by file path via importlib to avoid the `prompts` module-name clash with
6_evaluation/static_evaluation/prompts.py) so traced inputs match training.

Hypothesis (expected-behavior matrix):
  adult + ED + vitals                                -> ESI ok
  child <5 + low-resource + vitals                   -> IMCI/ETAT (no ESI)
  child <5 + low-resource + user explicitly asks ESI -> still IMCI/ETAT (no ESI)
  benign child + low-resource                        -> no ESI
The leak = the model emitting ESI for an under-5 low-resource case.

The clean-IMCI contrast is obtained at constant input from a naturally clean
generation of `esi_leak_resp` (see generate_outputs.py) — never a steered prompt.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path

_P = (Path(__file__).resolve().parent.parent
      / "6_evaluation" / "static_evaluation" / "prompts.py")
_spec = _ilu.spec_from_file_location("triage_static_prompts", str(_P))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SYSTEM_PROMPT = _mod.SYSTEM_PROMPT

PROMPTS: dict[str, dict] = {
    "esi_leak_resp": {
        "user": ("My 2-year-old has been making a harsh noise when breathing in "
                 "even when he is calm and resting, and the skin under his ribs "
                 "pulls in with every breath. We are hours from any hospital. "
                 "What should we do?"),
        "expected": "IMCI_ETAT",
        "note": "under-5 low-resource respiratory danger sign (stridor + chest indrawing)",
    },
    "adult_ed_resp": {
        "user": ("Emergency department: 58-year-old man, acute shortness of "
                 "breath, respiratory rate 32, oxygen saturation 89% on room "
                 "air, known COPD. What triage level and next step?"),
        "expected": "ESI",
        "note": "adult ED respiratory — ESI is appropriate (positive control)",
    },
    "benign_peds": {
        "user": ("My 3-year-old has had a mild fever since yesterday but is "
                 "drinking, eating and playing normally. We live in a rural "
                 "village far from a clinic. Do we need to travel to the hospital?"),
        "expected": "IMCI_ETAT_HOMECARE",
        "note": "benign pediatric low-resource — no ESI, no over-triage (negative control)",
    },
    "mietic_vitals": {
        "user": ("Patient: 2 years old. Complaint: difficulty breathing. Vitals: "
                 "HR 150, RR 55, SpO2 90%, T 38.9C. Please give the ESI level and "
                 "the predicted number of ED resources for this visit."),
        "expected": "IMCI_ETAT",
        "note": "under-5 MIETIC-style (full vitals) + explicit ESI request — maximal leak trigger",
    },
}

# Phrase sets scored as COMPLETE strings (sequence logprob), never single tokens.
ESI_PHRASES = [
    "ESI Level 2",
    "Emergency Severity Index",
    "high-risk criteria",
    "ED visit",
    "predicted number of resources",
]
IMCI_PHRASES = [
    "IMCI/ETAT",
    "WHO IMCI",
    "danger sign",
    "nearest health facility",
    "no ESI or SATS code",
]

# Substrings to DETECT framework in generated text. Action wording excluded; and
# bare "esi "/"sats " are NOT used (they false-match the round-5 explicit refusal
# "No ESI or SATS code is assigned", which is CLEAN, not a leak).
ESI_MARKERS = ["esi level", "emergency severity index", "predicted number of resources",
               "anticipated resources", "sats red", "sats orange", "sats yellow",
               "sats green"]
IMCI_MARKERS = ["imci", "etat", "danger sign"]

EXPECTED_MATRIX = {
    "adult_ed_resp": "ESI ok",
    "esi_leak_resp": "IMCI/ETAT (no ESI)",
    "mietic_vitals": "IMCI/ETAT (no ESI), even though ESI is requested",
    "benign_peds": "no ESI (home care / routine)",
}


def chat_messages(prompt_id: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPTS[prompt_id]["user"]},
    ]


__all__ = ["SYSTEM_PROMPT", "PROMPTS", "ESI_PHRASES", "IMCI_PHRASES",
           "ESI_MARKERS", "IMCI_MARKERS", "EXPECTED_MATRIX", "chat_messages"]
