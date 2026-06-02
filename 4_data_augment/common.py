"""Shared helpers for TriageLLM Phase 1 step 4 (data augmentation).

Reuses VLLMBatchClient + write_jsonl from step 2 via sibling import.
Provides:
  - JSONL I/O
  - stratified train/validation split
  - deterministic oversampling with tag annotation
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REWRITE_DIR = DATA_DIR / "rewrite"
AUGMENT_DIR = DATA_DIR / "augment"


_cleaning_candidates = sorted(REPO_ROOT.glob("*_data_cleaning"))
if not _cleaning_candidates:
    raise FileNotFoundError(
        f"no sibling '*_data_cleaning' folder under {REPO_ROOT}"
    )
_step2_path = _cleaning_candidates[-1] / "common.py"
_module_name = "triagellm_step2_common"
_spec = _ilu.spec_from_file_location(_module_name, str(_step2_path))
_step2 = _ilu.module_from_spec(_spec)
sys.modules[_module_name] = _step2
_spec.loader.exec_module(_step2)

VLLMBatchClient = _step2.VLLMBatchClient
VLLMConfig = _step2.VLLMConfig
setup_logging = _step2.setup_logging
write_jsonl = _step2.write_jsonl


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


_ACTION_PREFIX = "Action:"


def prepend_triage_label(records: list[dict]) -> int:
    """Ensure every record with a known Action leads with an `Action:` line.

    Phase 2 under-triage fix (Action-first): the disposition must be on line 1
    so it is independent of response length and never preceded by diagnostic
    resources. Step 3 `build_hybrid_answer` already emits Action-first for
    MIETIC cases and the authored seeds are written Action-first; this pass is a
    metadata-driven safety net that prepends the Action line to any labeled
    record still missing it. It NEVER fabricates an Action for records without
    one in metadata (guideline sections, identity), and NEVER forces an ESI/SATS
    code. Idempotent. Returns the number of records modified.
    """
    n_modified = 0
    for r in records:
        messages = r.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            continue
        content = messages[-1].get("content", "")
        if content.lstrip().startswith(_ACTION_PREFIX):
            continue
        action = (r.get("metadata") or {}).get("action")
        if not action:
            continue
        messages[-1]["content"] = f"{_ACTION_PREFIX} {action}\n\n{content}"
        n_modified += 1
    return n_modified


def length_filter(records: list[dict], max_chars: int = 4000) -> tuple[list[dict], int]:
    """Drop records whose assistant content exceeds max_chars.

    Returns (kept, dropped_count).
    """
    kept = []
    dropped = 0
    for r in records:
        assistant = r["messages"][-1]["content"]
        if len(assistant) > max_chars:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def _stratum_key(record: dict) -> tuple:
    m = record.get("metadata", {})
    return (m.get("doc_type"), m.get("style"), m.get("source"))


def stratify_split(
    records: list[dict],
    val_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified split by (doc_type, style, source).

    Annotates each record with metadata.split = train|validation.
    Returns (train, validation) with originals only — caller oversamples train.
    """
    rng = random.Random(seed)
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by_stratum[_stratum_key(r)].append(r)

    train: list[dict] = []
    val: list[dict] = []
    for stratum, group in by_stratum.items():
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_frac))) if len(group) >= 20 else 0
        for r in group[:n_val]:
            r2 = json.loads(json.dumps(r))
            r2.setdefault("metadata", {})["split"] = "validation"
            r2["metadata"]["oversample_tag"] = "original"
            val.append(r2)
        for r in group[n_val:]:
            r2 = json.loads(json.dumps(r))
            r2.setdefault("metadata", {})["split"] = "train"
            r2["metadata"]["oversample_tag"] = "original"
            train.append(r2)
    return train, val


def oversample(records: list[dict], factor: int) -> list[dict]:
    """Duplicate each record (factor - 1) times, tagging copies.

    Factor 1 = pass-through (original kept). Factor 10 = 1 original + 9 copies.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return list(records)
    out: list[dict] = []
    for r in records:
        out.append(r)
        for n in range(2, factor + 1):
            copy = json.loads(json.dumps(r))
            copy["metadata"]["oversample_tag"] = f"copy_{n}"
            copy["id"] = f"{r['id']}__x{n}"
            out.append(copy)
    return out


def shuffle_deterministic(records: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = list(records)
    rng.shuffle(out)
    return out


__all__ = [
    "REPO_ROOT", "DATA_DIR", "REWRITE_DIR", "AUGMENT_DIR",
    "VLLMBatchClient", "VLLMConfig", "setup_logging",
    "load_jsonl", "write_jsonl",
    "length_filter", "stratify_split", "oversample", "shuffle_deterministic",
    "prepend_triage_label",
]
