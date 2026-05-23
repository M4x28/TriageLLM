"""Merge a LoRA adapter into the base model and save as a full HF checkpoint.

Usage:
  python 5_sft_training/merge_lora.py --model qwen3-1.7b
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import CHECKPOINT_DIR, MODELS, setup_logging


def main() -> int:
    log = setup_logging("triagellm.sft.merge")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()

    spec = MODELS[args.model]
    adapter_dir = CHECKPOINT_DIR / spec.slug / "adapter"
    merged_dir = CHECKPOINT_DIR / spec.slug / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        log.error("adapter not found at %s — run train.py first", adapter_dir)
        return 1

    log.info("loading base %s + adapter %s", spec.hf_id, adapter_dir)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.needs_trust_remote_code,
    )
    base = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=spec.needs_trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    log.info("merging adapter into base weights")
    merged = peft_model.merge_and_unload()

    log.info("saving merged checkpoint to %s", merged_dir)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
