"""Shared helpers for TriageLLM step 6 evaluation.

Avoids code duplication by importing:
  * model registry  -> sibling step 5 (via importlib)
  * setup_logging   -> sibling step 2 (chained through step 5)
  * write_jsonl     -> sibling step 2

Adds locally:
  * load_jsonl(path)         tiny JSONL reader used by qualitative + summary
  * load_merged_model(spec)  load a step-5 merged HF checkpoint in bf16
  * generate(model, tokenizer, system, user, max_new_tokens)
      single greedy generation with apply_chat_template + safe fallback
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUGMENT_DIR = DATA_DIR / "augment"
SFT_DIR = DATA_DIR / "sft"
CHECKPOINT_DIR = SFT_DIR / "checkpoints"
EVAL_DIR = DATA_DIR / "eval"
INTERNAL_DIR = EVAL_DIR / "internal"
EXTERNAL_DIR = EVAL_DIR / "external"
QUALITATIVE_DIR = EVAL_DIR / "qualitative"


def _load_sibling_module(folder_glob: str, name: str):
    """Load a Python module from a sibling step folder by glob."""
    candidates = sorted(REPO_ROOT.glob(folder_glob))
    if not candidates:
        raise FileNotFoundError(f"no sibling '{folder_glob}' folder under {REPO_ROOT}")
    path = candidates[-1] / "common.py"
    spec = _ilu.spec_from_file_location(name, str(path))
    module = _ilu.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Step 5 module: gives us the canonical ModelSpec / MODELS registry.
_step5 = _load_sibling_module("*_sft_training", "triagellm_step5_common")
ModelSpec = _step5.ModelSpec
MODELS = _step5.MODELS                  # qwen3-1.7b / smollm3-3b / gemma-3n-e2b
setup_logging = _step5.setup_logging
write_jsonl = _step5.write_jsonl


# Reference model for the "out of competition" external comparison.
REFERENCE_MODEL_SLUG = "medgemma-4b"
REFERENCE_MODEL_SPEC = ModelSpec(
    slug=REFERENCE_MODEL_SLUG,
    hf_id="google/medgemma-4b-it",
    per_device_batch=1,
    lora_target_modules=[],
    needs_trust_remote_code=False,
    notes="Reference upper-bound: medical-pretrained 4B (gated, HF token needed).",
)


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Used by qualitative + summary."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_checkpoint(spec: ModelSpec, *,
                       allow_reference: bool = False,
                       baseline: bool = False) -> str:
    """Return the path or HF id to load weights from.

    For the 3 SFT candidates: prefer the merged checkpoint produced by step 5.
    For the reference model (MedGemma): always load from HF id.
    For Phase 2 `baseline=True` runs: load the base model directly from HF id
    (no merged checkpoint required) so we can probe pre-SFT capabilities.
    """
    if baseline:
        return spec.hf_id
    if spec.slug == REFERENCE_MODEL_SLUG:
        if not allow_reference:
            raise ValueError("reference model must be opted in explicitly")
        return spec.hf_id
    merged = CHECKPOINT_DIR / spec.slug / "merged"
    if merged.exists():
        return str(merged)
    raise FileNotFoundError(
        f"merged checkpoint missing for {spec.slug} at {merged}: "
        "run step 5 train.py + merge_lora.py first"
    )


def load_merged_model(spec: ModelSpec, *,
                      allow_reference: bool = False,
                      baseline: bool = False,
                      use_4bit: bool = False):
    """Load tokenizer + model on the visible CUDA device.

    use_4bit=True: BnB NF4 4-bit (27B → ~14 GB, fits single L40S).
    Faster inference than BF16 multi-GPU pipeline parallelism.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    source = resolve_checkpoint(spec, allow_reference=allow_reference,
                                baseline=baseline)
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=spec.needs_trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=spec.needs_trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=spec.needs_trust_remote_code,
        )
    model.train(False)
    return tokenizer, model


def generate(model, tokenizer, system: str, user: str,
             max_new_tokens: int = 256) -> tuple[str, float]:
    """Greedy generation with chat template + measured wall-clock latency.

    Returns (decoded_text, elapsed_seconds). The caller can derive tok/s as
    `(max_new_tokens / elapsed)` (upper-bound estimate, since the model may
    emit fewer than max_new_tokens before EOS).
    """
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        # enable_thinking=False suppresses Qwen3 chain-of-thought traces.
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = f"<system>{system}</system>\n<user>{user}</user>\n<assistant>"

    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    generated = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, elapsed


__all__ = [
    "REPO_ROOT", "DATA_DIR", "AUGMENT_DIR", "SFT_DIR", "CHECKPOINT_DIR",
    "EVAL_DIR", "INTERNAL_DIR", "EXTERNAL_DIR", "QUALITATIVE_DIR",
    "MODELS", "ModelSpec", "REFERENCE_MODEL_SLUG", "REFERENCE_MODEL_SPEC",
    "setup_logging", "write_jsonl", "load_jsonl",
    "resolve_checkpoint", "load_merged_model", "generate",
]
