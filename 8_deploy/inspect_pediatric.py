"""Dump 5 pediatric IMCI/ETAT records with gold keywords and Q4_K_M response.

Used to diagnose why pediatric recall dropped -35pt vs BF16. Prints
side-by-side: user prompt, gold answer, gold keywords found, Q4_K_M
response, matched vs missed keywords.

Usage:
  python 7_deploy/inspect_pediatric.py \
      --gguf data/deploy/gguf/qwen3-1.7b-q4_k_m.gguf
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    PEDIATRIC_CONTENT_KEYWORDS,
    generate_gguf,
    load_gguf_model,
    load_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--validation-jsonl",
                    default=str(REPO_ROOT / "data" / "augment" / "validation.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--n-scan", type=int, default=400)
    args = ap.parse_args()

    records = load_jsonl(Path(args.validation_jsonl))
    pedi = [r for r in records[:args.n_scan]
            if r.get("metadata", {}).get("source") in ("who_imci", "who_etat")]
    print(f"=== {len(pedi)} pediatric records (IMCI/ETAT) ===")

    llama = load_gguf_model(Path(args.gguf), n_ctx=4096, n_gpu_layers=0,
                            logits_all=False)

    for i, rec in enumerate(pedi):
        msgs = rec["messages"]
        gold_text = msgs[2]["content"]
        gold_lower = gold_text.lower()
        gold_hits = [kw for kw in PEDIATRIC_CONTENT_KEYWORDS if kw in gold_lower]

        text, dt, _ = generate_gguf(
            llama, msgs[0]["content"], msgs[1]["content"],
            max_new_tokens=args.max_new_tokens,
        )
        text_lower = text.lower()
        matched = [kw for kw in gold_hits if kw in text_lower]
        missed = [kw for kw in gold_hits if kw not in text_lower]

        meta = rec.get("metadata", {})
        print(f"\n--- record {i+1} (source={meta.get('source')}, "
              f"style={meta.get('style', '?')}, "
              f"task={meta.get('task_type', '?')}) ---")
        print(f"USER: {msgs[1]['content'][:250]}")
        print(f"GOLD answer preview: {gold_text[:300]}")
        print(f"GOLD keywords found ({len(gold_hits)}): {gold_hits}")
        print(f"Q4KM matched ({len(matched)}/{len(gold_hits)} "
              f"= {len(matched)/max(len(gold_hits),1):.2f}): {matched}")
        print(f"Q4KM MISSED: {missed}")
        print(f"Q4KM resp ({len(text)} chars, {dt:.1f}s):")
        print(f"  {text[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
