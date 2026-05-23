"""Qualitative spot-check: 30 stratified samples per model for manual review.

Picks up to 5 records per `metadata.style` (6 styles in step 4 dataset)
and records the model generation alongside the gold answer. Output JSONL
is meant to be inspected by a human reader; nothing is scored
automatically here.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 6_evaluation/eval_qualitative.py --model qwen3-1.7b
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import (
    AUGMENT_DIR,
    MODELS,
    QUALITATIVE_DIR,
    generate,
    load_jsonl,
    load_merged_model,
    setup_logging,
    write_jsonl,
)


def _stratified_sample(records: list[dict], per_style: int,
                       seed: int) -> list[dict]:
    """Pick up to `per_style` records for each metadata.style bucket."""
    rng = random.Random(seed)
    by_style: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        style = r.get("metadata", {}).get("style", "?")
        by_style[style].append(r)
    out: list[dict] = []
    for style in sorted(by_style):
        group = by_style[style]
        rng.shuffle(group)
        out.extend(group[:per_style])
    return out


def main() -> int:
    log = setup_logging("triagellm.eval.qualitative")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--validation-jsonl",
                    default=str(AUGMENT_DIR / "validation.jsonl"))
    ap.add_argument("--per-style", type=int, default=5,
                    help="how many records to sample per metadata.style")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = MODELS[args.model]
    QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QUALITATIVE_DIR / f"{spec.slug}_samples.jsonl"
    log.info("eval_qualitative model=%s -> %s", spec.slug, out_path)

    records = load_jsonl(Path(args.validation_jsonl))
    sample = _stratified_sample(records, args.per_style, args.seed)
    log.info("sampled %d records across %d styles", len(sample),
             len({r.get('metadata', {}).get('style') for r in sample}))

    tokenizer, model = load_merged_model(spec)

    rows: list[dict] = []
    for rec in sample:
        messages = rec["messages"]
        gold_assistant = messages[2]["content"]
        text, dt = generate(model, tokenizer,
                            messages[0]["content"], messages[1]["content"],
                            max_new_tokens=1024)
        rows.append({
            "id": rec.get("id"),
            "metadata": rec.get("metadata", {}),
            "user_prompt": messages[1]["content"],
            "gold_answer": gold_assistant,
            "model_response": text,
            "latency_s": round(dt, 2),
        })

    n_written = write_jsonl(out_path, rows)
    log.info("wrote %d qualitative samples to %s", n_written, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
