"""Shared helpers for TriageLLM Phase 1 step 5 (SFT training).

Reuses path constants and JSONL I/O via sibling import from step 2.
Provides:
  - path constants (REPO_ROOT, AUGMENT_DIR, SFT_DIR, etc.)
  - model registry with HF id + slug + arch quirks
  - dataset loader for HF messages JSONL → datasets.Dataset
  - logging setup
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUGMENT_DIR = DATA_DIR / "augment"
SFT_DIR = DATA_DIR / "sft"
BASELINE_DIR = SFT_DIR / "baselines"
CHECKPOINT_DIR = SFT_DIR / "checkpoints"
GGUF_DIR = SFT_DIR / "gguf"
EVAL_DIR = SFT_DIR / "eval"

_cleaning_candidates = sorted(REPO_ROOT.glob("*_data_cleaning"))
if not _cleaning_candidates:
    raise FileNotFoundError(f"no sibling '*_data_cleaning' folder under {REPO_ROOT}")
_step2_path = _cleaning_candidates[-1] / "common.py"
_module_name = "triagellm_step2_common"
_spec = _ilu.spec_from_file_location(_module_name, str(_step2_path))
_step2 = _ilu.module_from_spec(_spec)
sys.modules[_module_name] = _step2
_spec.loader.exec_module(_step2)

setup_logging = _step2.setup_logging
write_jsonl = _step2.write_jsonl


@dataclass
class ModelSpec:
    slug: str               # short id for output paths
    hf_id: str              # HuggingFace repo
    per_device_batch: int   # batch size capped by VRAM @ 46GB L40S
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    needs_trust_remote_code: bool = False
    chat_template: str = "auto"  # transformers picks via tokenizer config
    # QLoRA: load base in 4-bit NF4 to fit 27B+ on single L40S 46 GB
    use_4bit_base: bool = False
    # 8-bit INT8: more stable than NF4 for MoE architectures; 35B = 35 GB
    use_8bit_base: bool = False
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    # ----- Phase 1 candidates (kept for backward compatibility) -----
    "qwen3-1.7b": ModelSpec(
        slug="qwen3-1.7b",
        hf_id="Qwen/Qwen3-1.7B",
        per_device_batch=4,
        notes="Phase 1 primary. ChatML, GQA 16/8, thinking off via tokenizer kwarg.",
    ),
    "qwen3-1.7b-phase1": ModelSpec(
        slug="qwen3-1.7b-phase1",
        hf_id="Qwen/Qwen3-1.7B",
        per_device_batch=4,
        notes="Phase 1 primary checkpoint, preserved for the Phase 2 before/after "
              "comparison: re-evaluated on the corrected validation gold.",
    ),
    "smollm3-3b": ModelSpec(
        slug="smollm3-3b",
        hf_id="HuggingFaceTB/SmolLM3-3B",
        per_device_batch=2,
        notes="Phase 1. Apache, 11.2T train, IT support, NoPE 3:1.",
    ),
    "gemma-3n-e2b": ModelSpec(
        slug="gemma-3n-e2b",
        hf_id="google/gemma-3n-E2B-it",
        per_device_batch=2,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        notes="Phase 1. MatFormer + Per-Layer Embeddings; LoRA only on attention.",
    ),

    # ----- Phase 2 candidates (mid tier, mobile 6-8 GB RAM target) -----
    "qwen3-8b": ModelSpec(
        slug="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        per_device_batch=2,
        notes="Phase 2 mid M1. 4.7x scaling vs Phase 1 primary. Q4_K_M ~4.5 GB.",
    ),
    "phi-4-mini": ModelSpec(
        slug="phi-4-mini",
        hf_id="microsoft/Phi-4-mini-instruct",
        per_device_batch=4,
        # Phi-4 uses merged qkv_proj + gate_up_proj. Verify target_modules at SFT time.
        lora_target_modules=[
            "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
        ],
        notes="Phase 2 mid M2. MIT, 22 languages incl. FR/AR/PT/IT, 128K ctx.",
    ),
    "medgemma-1.5-4b": ModelSpec(
        slug="medgemma-1.5-4b",
        hf_id="google/medgemma-1.5-4b-it",
        per_device_batch=4,
        notes="Phase 2 mid M3. Medical-pretrained SOTA Jan 2026 (91% MedQA). Multimodal text+image.",
    ),

    # ----- Phase 2 candidates (large tier, server inference) -----
    "qwen3.6-27b": ModelSpec(
        slug="qwen3.6-27b",
        hf_id="Qwen/Qwen3.6-27B",
        per_device_batch=1,
        use_4bit_base=True,
        notes="Phase 2 server S1. Apr 2026 flagship dense. QLoRA NF4 on L40S 46GB.",
    ),
    "qwen3.6-35b-moe": ModelSpec(
        slug="qwen3.6-35b-moe",
        hf_id="Qwen/Qwen3.6-35B-A3B",
        per_device_batch=1,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        notes="Phase 2 server S2. MoE: 35B total / 3B active. BF16 LoRA on GPU 0+1 (70 GB).",
    ),
    "medgemma-27b": ModelSpec(
        slug="medgemma-27b",
        hf_id="google/medgemma-27b-text-it",
        per_device_batch=1,
        use_4bit_base=True,
        notes="Phase 2 large L3. Medical 27B text-only it. QLoRA NF4 on L40S 46GB.",
    ),
}


def load_hf_messages_dataset(jsonl_path: Path):
    """Load a HF messages JSONL into datasets.Dataset.

    Each record must have {messages: [{role, content}, ...], metadata}.
    The dataset retains only the `messages` field — SFTTrainer applies
    the chat template at tokenization time.
    """
    from datasets import Dataset
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"messages": r["messages"]})
    return Dataset.from_list(rows)


__all__ = [
    "REPO_ROOT", "DATA_DIR", "AUGMENT_DIR", "SFT_DIR",
    "BASELINE_DIR", "CHECKPOINT_DIR", "GGUF_DIR", "EVAL_DIR",
    "ModelSpec", "MODELS",
    "load_hf_messages_dataset",
    "setup_logging", "write_jsonl",
]
