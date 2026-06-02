"""Shared helpers for TriageLLM Phase 1 step 3 (Q&A rewrite).

Reuses VLLMBatchClient + path helpers from step 2 via sibling import.
Provides:
  - envelope JSONL loader
  - HF messages record builder
  - triage label derivation (task_type → ESI/SATS)
  - JSONL writer (re-exported)
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REWRITE_DIR = DATA_DIR / "rewrite"

# Reuse helpers from step 2 via explicit file load (avoid name collision
# with this module, also named common.py). Resolve the folder by glob so a
# step renumbering (e.g. 3_data_cleaning -> 2_data_cleaning) does not break it.
import importlib.util as _ilu
_cleaning_candidates = sorted(REPO_ROOT.glob("*_data_cleaning"))
if not _cleaning_candidates:
    raise FileNotFoundError(f"no sibling '*_data_cleaning' folder under {REPO_ROOT}")
_step2_path = _cleaning_candidates[-1] / "common.py"
_module_name = "triagellm_step2_common"
_spec = _ilu.spec_from_file_location(_module_name, str(_step2_path))
_step2 = _ilu.module_from_spec(_spec)
sys.modules[_module_name] = _step2  # required for dataclasses inside module
_spec.loader.exec_module(_step2)

VLLMBatchClient = _step2.VLLMBatchClient
VLLMConfig = _step2.VLLMConfig
setup_logging = _step2.setup_logging
write_jsonl = _step2.write_jsonl


# ─────────────────────────────────────────────────────────────────────────────
# Envelope loader
# ─────────────────────────────────────────────────────────────────────────────

def load_envelope(path: Path) -> list[dict]:
    """Read JSONL produced by step 2 (envelope schema)."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_envelope(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ─────────────────────────────────────────────────────────────────────────────
# Triage label derivation (MIETIC task answer → ESI/SATS)
# ─────────────────────────────────────────────────────────────────────────────
#
# Phase 2 fix. The original code mapped task_type directly to ESI
# (esi1_detection -> ESI 1), which is wrong: MIETIC tasks are *decision points*
# of the ESI algorithm, not the patient's level. `esi1_detection` asks "does
# this patient need a life-saving intervention?" — the answer can be yes or no.
# The old logic labeled every esi1_detection case ESI 1 / SATS Red, collapsing
# the whole corpus onto a single (wrong) label. We now read the answer polarity
# / resource count from the reasoning trace and apply the real ESI tree.

ESI_TO_SATS = {
    1: "Red (immediate)",
    2: "Orange (very urgent)",
    3: "Yellow (urgent)",
    4: "Green (less urgent)",
    5: "Green (non-urgent)",
}

# Standard ESI -> Action mapping (Phase 2 under-triage fix). The Action line is
# separate from the ESI severity code: the model must lead every clinical answer
# with an Action so "escalate" is no longer conflated with "assign ESI 2".
ESI_TO_ACTION = {
    1: "REFER NOW",
    2: "URGENT SAME-DAY CARE",
    3: "URGENT SAME-DAY CARE",
    4: "ROUTINE FOLLOW-UP",
    5: "HOME CARE + RETURN ADVICE",
}

# A clear "this patient needs a life-saving / immediate intervention" verdict in
# an esi1_detection trace => ESI 1. Anything weaker (negative or unclear) is not
# labeled from this task alone: the true level is 2-5 and undeterminable here.
_ESI1_POSITIVE = re.compile(
    r"(requires?|require[sd]|needs?|necessitat\w+|warrant\w+|indicat\w+)"
    r"[^.]{0,60}?(immediate\s+)?"
    r"(life[\s-]?saving|life[\s-]?threatening|resuscitat\w+|"
    r"intubation|emergent\s+airway|immediate\s+intervention)"
    r"|is\s+critically\s+ill|peri[\s-]?arrest|in\s+extremis",
    re.IGNORECASE,
)
_ESI1_NEGATIVE = re.compile(
    r"(does\s+not|doesn'?t|no\s+immediate|not\s+require|without\s+the\s+need|"
    r"does\s+not\s+(require|need|meet))"
    r"[^.]{0,60}?(life[\s-]?saving|immediate\s+inter|resuscitat)",
    re.IGNORECASE,
)

_WORD_NUM = {"zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4,
             "five": 5, "six": 6}
_RES_DIGIT = re.compile(r"(\d+)\s+resource", re.IGNORECASE)
_RES_WORD = re.compile(
    r"\b(zero|no|one|two|three|four|five|six)\s+resource", re.IGNORECASE)


def _esi1_is_positive(trace: str) -> bool | None:
    """True if the esi1 trace clearly affirms a life-saving need, False if it
    clearly denies it, None if unclear."""
    if _ESI1_NEGATIVE.search(trace):
        return False
    if _ESI1_POSITIVE.search(trace):
        return True
    return None


def _parse_resource_count(trace: str) -> int | None:
    """Parse the predicted ED resource count from a resource_prediction trace.

    Returns the last count mentioned, or None if not parseable.
    """
    nums = _RES_DIGIT.findall(trace)
    if nums:
        return int(nums[-1])
    words = _RES_WORD.findall(trace)
    if words:
        return _WORD_NUM[words[-1].lower()]
    return None


def _vitals_danger_zone(patient: dict) -> bool:
    """ESI danger-zone vital signs (age-stratified). Upgrades a resource-based
    ESI 3 to ESI 2 per the ESI handbook tie-breaker."""
    vs = (patient or {}).get("vital_signs") or {}
    age = (patient or {}).get("age_years")
    hr = vs.get("heart_rate")
    rr = vs.get("resp_rate")
    spo2 = vs.get("spo2")
    if spo2 is not None and spo2 < 92:
        return True
    if age is None:
        hr_max, rr_max = 100, 20
    elif age >= 8:
        hr_max, rr_max = 100, 20
    elif age >= 3:
        hr_max, rr_max = 140, 30
    elif age >= 1:
        hr_max, rr_max = 160, 40
    else:
        hr_max, rr_max = 180, 50
    if hr is not None and hr > hr_max:
        return True
    if rr is not None and rr > rr_max:
        return True
    return False


@dataclass
class TriageLabel:
    """Structured triage label for a clinical record.

    `framework` is one of ESI | IMCI_ETAT | SATS | OUT_OF_SCOPE. `action` is the
    closed-vocabulary Action line. `esi`/`sats` are populated only for the ESI
    framework (MIETIC adult ED cases).
    """
    framework: str
    action: str
    esi: int | None = None
    sats: str | None = None


def _derive_esi(case: dict) -> int | None:
    """Compute the ESI level from a MIETIC task answer, or None if not derivable.

    `esi1_detection` only yields ESI 1 on a positive verdict; a negative/unclear
    verdict rules out ESI 1 but leaves the true level (2-5) undeterminable here.
    `resource_prediction` maps the predicted resource count to ESI 3/4/5, with a
    danger-zone-vitals tie-breaker upgrade to ESI 2.
    """
    task = case.get("task_type")
    trace = case.get("reasoning_trace", "") or ""
    if task == "esi1_detection":
        return 1 if _esi1_is_positive(trace) is True else None
    if task == "resource_prediction":
        n = _parse_resource_count(trace)
        if n is None:
            return None
        esi = 3 if n >= 2 else (4 if n == 1 else 5)
        if _vitals_danger_zone(case.get("patient", {})):
            esi = 2
        return esi
    return None


def derive_triage(case: dict) -> TriageLabel | None:
    """Derive the structured triage label (framework + Action + ESI/SATS) from a
    MIETIC case, or None when the level cannot be determined from this task.

    MIETIC is adult ED data, so a derivable case is always the ESI framework.
    Pediatric IMCI/ETAT and out-of-domain labels come from authored seeds and
    the danger-sign labeler, not from this function.
    """
    if not case:
        return None
    esi = _derive_esi(case)
    if esi is None:
        return None
    return TriageLabel(
        framework="ESI",
        action=ESI_TO_ACTION[esi],
        esi=esi,
        sats=ESI_TO_SATS.get(esi, ""),
    )


def derive_triage_label(case: dict) -> str | None:
    """Human-readable "ESI N / SATS Color (...)" line, or None if not derivable.

    Thin formatter over `derive_triage`, kept for callers that only need the
    label string.
    """
    t = derive_triage(case)
    return f"ESI {t.esi} / SATS {t.sats}" if t and t.esi is not None else None


def triage_metadata(case: dict) -> dict:
    """Triage fields to merge into a record's metadata, or {} if not derivable."""
    t = derive_triage(case)
    if not t:
        return {}
    return {"triage_framework": t.framework, "action": t.action,
            "esi": t.esi, "sats": t.sats}


# ─────────────────────────────────────────────────────────────────────────────
# HF messages record builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MessagesRecord:
    id: str
    messages: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": self.messages,
            "metadata": self.metadata,
        }


def build_messages_record(
    record_id: str,
    system_prompt: str,
    question: str,
    answer: str,
    metadata: dict,
) -> dict:
    return MessagesRecord(
        id=record_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        metadata=metadata,
    ).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid answer construction (MIETIC)
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_answer(case: dict, reasoning_trace: str) -> str:
    """Build the assistant answer: Action line first, then reasoning, then the
    ESI/SATS code footer.

    Action-first (Phase 2 under-triage fix) puts the disposition on line 1 and
    keeps any resource/diagnostic mentions inside the reasoning body strictly
    after the Action line. If no label is derivable (e.g. task_type=other), the
    reasoning trace is returned unchanged.
    """
    body = (reasoning_trace or "").strip()
    t = derive_triage(case)
    if not t:
        return body
    parts = [f"Action: {t.action}"]
    if body:
        parts += ["", body]
    if t.framework == "ESI":
        parts += ["", "---", f"**Triage**: ESI {t.esi} / SATS {t.sats}"]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL writer with skip-record counter
# ─────────────────────────────────────────────────────────────────────────────

def write_messages_jsonl(path: Path, records: list[dict]) -> int:
    return write_jsonl(path, records)


# Re-export
__all__ = [
    "REPO_ROOT", "DATA_DIR", "PROCESSED_DIR", "REWRITE_DIR",
    "VLLMBatchClient", "VLLMConfig", "setup_logging",
    "load_envelope", "iter_envelope",
    "ESI_TO_ACTION", "TriageLabel", "derive_triage", "derive_triage_label",
    "triage_metadata", "build_hybrid_answer",
    "build_messages_record", "MessagesRecord",
    "write_messages_jsonl", "write_jsonl",
]
