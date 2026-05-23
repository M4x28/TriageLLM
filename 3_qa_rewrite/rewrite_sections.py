"""Rewrite PDF guideline sections (envelope JSONL) → 3-style HF messages JSONL.

Per record × 3 styles (direct_question, protocol_lookup, red_flag_check)
= 1 dedicated LLM call per style. SKIP filter drops TOC/junk sections
based on LLM judgment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    PROCESSED_DIR,
    REWRITE_DIR,
    VLLMBatchClient,
    VLLMConfig,
    build_messages_record,
    iter_envelope,
    setup_logging,
    write_messages_jsonl,
)
from prompts import (
    PROMPT_VERSION,
    QUESTION_SCHEMA,
    STYLES_FOR_SECTIONS,
    SYSTEM_PROMPT,
    render_prompt,
)


SOURCES = ("sats", "imci", "etat")
SRC_FILES = {
    "sats": "sats_sections.jsonl",
    "imci": "imci_sections.jsonl",
    "etat": "etat_sections.jsonl",
}
OUT_PATH = REWRITE_DIR / "train_sections.jsonl"


def build_section_answer(rec: dict) -> str:
    """Return the section text as the assistant answer.

    Prefers `section.text` (clean body) over `raw_text` (may include header).
    """
    section = rec.get("section") or {}
    text = (section.get("text") or rec.get("raw_text") or "").strip()
    return text


def main() -> int:
    log = setup_logging("triagellm.rewrite.sections")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES) + ["all"], default="all")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    sources = SOURCES if args.source == "all" else (args.source,)

    # Load all envelope records, track origin
    records: list[dict] = []
    for src in sources:
        path = PROCESSED_DIR / SRC_FILES[src]
        if not path.exists():
            log.warning("missing %s — skip", path)
            continue
        recs = list(iter_envelope(path))
        log.info("[%s] loaded %d records", src, len(recs))
        records.extend(recs)

    if not records:
        log.error("no source records found")
        return 1
    if args.limit:
        records = records[:args.limit]
        log.info("limited to %d records", len(records))

    client = VLLMBatchClient(VLLMConfig(model_name=args.model))

    # Pre-compute answers
    answers = [build_section_answer(r) for r in records]

    out_records: list[dict] = []
    skip_count_by_style: dict[str, int] = {}
    kept_count_by_style: dict[str, int] = {}

    for style in STYLES_FOR_SECTIONS:
        log.info("generating questions for style '%s' (%d records)",
                 style, len(records))
        prompts = [render_prompt(r["raw_text"], style) for r in records]
        extracted = client.batch_extract(
            prompts, QUESTION_SCHEMA,
            max_tokens=args.max_tokens,
            temperature=0.4,
        )

        skipped = 0
        kept = 0
        for rec, ext, answer in zip(records, extracted, answers):
            if ext.get("skip") is True:
                skipped += 1
                continue
            question = (ext.get("question") or "").strip()
            if not question:
                skipped += 1
                continue
            section = rec.get("section") or {}
            metadata = {
                "source_id": rec["id"],
                "source": rec["source"],
                "doc_type": rec["doc_type"],
                "style": style,
                "section_type": section.get("section_type"),
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
            kept += 1
        skip_count_by_style[style] = skipped
        kept_count_by_style[style] = kept
        log.info("style '%s' kept %d, skipped %d", style, kept, skipped)

    n = write_messages_jsonl(OUT_PATH, out_records)
    log.info("wrote %d records → %s", n, OUT_PATH)
    log.info("kept per style: %s", kept_count_by_style)
    log.info("skipped per style: %s", skip_count_by_style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
