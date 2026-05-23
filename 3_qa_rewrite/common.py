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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REWRITE_DIR = DATA_DIR / "rewrite"

# Reuse helpers from step 2 via explicit file load (avoid name collision
# with this module, also named common.py)
import importlib.util as _ilu
_step2_path = REPO_ROOT / "3_data_cleaning" / "common.py"
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
# Triage label derivation (MIETIC task_type → ESI/SATS)
# ─────────────────────────────────────────────────────────────────────────────

TASK_TO_ESI = {
    "esi1_detection": 1,
    "esi2_detection": 2,
    # resource_prediction does not map cleanly without parsing the output
}

ESI_TO_SATS = {
    1: "Red (immediate)",
    2: "Orange (very urgent)",
    3: "Yellow (urgent)",
    4: "Green (less urgent)",
    5: "Green (non-urgent)",
}


def derive_triage_label(case: dict) -> str | None:
    """Return human-readable ESI/SATS line or None if not derivable."""
    if not case:
        return None
    task = case.get("task_type")
    esi = TASK_TO_ESI.get(task)
    if esi is None:
        return None
    sats = ESI_TO_SATS.get(esi, "")
    return f"ESI {esi} / SATS {sats}"


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
    """Concat reasoning_trace with explicit triage label section.

    If no derivable label (e.g. task_type=other), return reasoning_trace
    unchanged.
    """
    label = derive_triage_label(case)
    body = (reasoning_trace or "").strip()
    if not label:
        return body
    return f"{body}\n\n---\n**Triage recommendation**: {label}"


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
    "derive_triage_label", "build_hybrid_answer",
    "build_messages_record", "MessagesRecord",
    "write_messages_jsonl", "write_jsonl",
]
