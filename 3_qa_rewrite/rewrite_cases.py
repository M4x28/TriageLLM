"""Rewrite MIETIC cases (envelope JSONL) → 3-style HF messages JSONL.

Per record × 3 styles (caregiver_query, clinician_handoff, field_worker_query)
= 1 dedicated LLM call per style. No multi-style batching for medical-domain
quality reasons (see docs/4_qa_rewrite.md).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common import (
    PROCESSED_DIR,
    REWRITE_DIR,
    VLLMBatchClient,
    VLLMConfig,
    build_hybrid_answer,
    build_messages_record,
    iter_envelope,
    setup_logging,
    write_messages_jsonl,
)
from prompts import (
    PROMPT_VERSION,
    QUESTION_SCHEMA,
    STYLES_FOR_CASES,
    SYSTEM_PROMPT,
    render_prompt,
)


SRC_PATH = PROCESSED_DIR / "mietic.jsonl"
OUT_PATH = REWRITE_DIR / "train_cases.jsonl"


def main() -> int:
    log = setup_logging("triagellm.rewrite.cases")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N records (debug)")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    if not SRC_PATH.exists():
        log.error("MIETIC envelope missing: %s — run step 2 first", SRC_PATH)
        return 1

    records = list(iter_envelope(SRC_PATH))
    log.info("loaded %d MIETIC envelope records", len(records))
    if args.limit:
        records = records[:args.limit]
        log.info("limited to %d records", len(records))

    client = VLLMBatchClient(VLLMConfig(model_name=args.model))

    # Pre-compute hybrid answers (deterministic, no LLM)
    hybrid_answers: list[str] = []
    for rec in records:
        case = rec.get("case") or {}
        rt = case.get("reasoning_trace", "")
        hybrid_answers.append(build_hybrid_answer(case, rt))

    out_records: list[dict] = []
    style_stats: dict[str, int] = {}

    for style in STYLES_FOR_CASES:
        log.info("generating questions for style '%s' (%d records)",
                 style, len(records))
        prompts = [render_prompt(r["raw_text"], style) for r in records]
        extracted = client.batch_extract(
            prompts, QUESTION_SCHEMA,
            max_tokens=args.max_tokens,
            temperature=0.4,
        )

        n_kept = 0
        for rec, ext, answer in zip(records, extracted, hybrid_answers):
            question = (ext.get("question") or "").strip()
            if not question:
                continue
            case = rec.get("case") or {}
            metadata = {
                "source_id": rec["id"],
                "source": rec["source"],
                "doc_type": rec["doc_type"],
                "style": style,
                "task_type": case.get("task_type"),
                "prompt_version": PROMPT_VERSION,
            }
            record_id = f"qa_{rec['id']}_{style}"
            out_records.append(build_messages_record(
                record_id=record_id,
                system_prompt=SYSTEM_PROMPT,
                question=question,
                answer=answer,
                metadata=metadata,
            ))
            n_kept += 1
        style_stats[style] = n_kept
        log.info("style '%s' kept %d/%d", style, n_kept, len(records))

    n = write_messages_jsonl(OUT_PATH, out_records)
    log.info("wrote %d records → %s", n, OUT_PATH)
    log.info("per-style counts: %s", style_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
