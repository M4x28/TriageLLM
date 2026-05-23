"""Clean MIETIC parquet → mietic.jsonl with unified envelope schema.

Flow:
  1. Load parquet from data/raw/mietic/
  2. Regex extract vitals + demographics + task_type
  3. vLLM batch extract chief_complaint + symptoms + history
  4. Build records, length filter, write JSONL
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    PROCESSED_DIR,
    RAW_DIR,
    Provenance,
    VLLMBatchClient,
    VLLMConfig,
    build_record,
    classify_task_type,
    length_filter,
    parse_demographics,
    parse_vitals,
    setup_logging,
    write_jsonl,
)
from prompts import (
    MIETIC_EXTRACT_SCHEMA,
    PROMPT_VERSION,
    render_mietic_prompt,
)


MIETIC_PARQUET = RAW_DIR / "mietic" / "train.parquet"
OUT_PATH = PROCESSED_DIR / "mietic.jsonl"
LICENSE = "CC-BY-NC-SA-4.0"
CITATION = "MIETIC (jackf7499/MIETIC, HuggingFace)"


def main() -> int:
    log = setup_logging("triagellm.clean.mietic")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=VLLMConfig.model_name)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N rows (debug)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip LLM extraction (regex-only output)")
    args = ap.parse_args()

    if not MIETIC_PARQUET.exists():
        log.error("MIETIC parquet missing: %s — run step 1 first", MIETIC_PARQUET)
        return 1

    log.info("loading %s", MIETIC_PARQUET)
    df = pd.read_parquet(MIETIC_PARQUET)
    log.info("%d rows", len(df))
    if args.limit:
        df = df.head(args.limit)

    valid_rows = []
    for idx, row in df.iterrows():
        raw_input = row.get("input", "") or ""
        if not length_filter(raw_input, min_chars=50, max_chars=20_000):
            continue
        valid_rows.append((int(idx), row.to_dict()))

    log.info("%d rows pass length filter", len(valid_rows))

    llm_results: list[dict] = []
    if not args.skip_llm:
        client = VLLMBatchClient(VLLMConfig(model_name=args.model))
        prompts = [render_mietic_prompt(r["input"]) for _, r in valid_rows]
        log.info("submitting %d prompts to vLLM (batch_size=%d)",
                 len(prompts), args.batch_size)
        llm_results = client.batch_extract(prompts, MIETIC_EXTRACT_SCHEMA)
    else:
        llm_results = [{} for _ in valid_rows]

    records = []
    for (idx, row), extracted in zip(valid_rows, llm_results):
        raw_input = row.get("input", "") or ""
        raw_output = row.get("output", "") or ""
        instruction = row.get("instruction", "") or ""

        vitals = parse_vitals(raw_input)
        demo = parse_demographics(raw_input)
        task_type = classify_task_type(instruction)

        case = {
            "task_type": task_type,
            "patient": {
                **demo,
                "vital_signs": vitals,
                "chief_complaint": extracted.get("chief_complaint", ""),
                "symptoms": extracted.get("symptoms", []),
                "history": extracted.get("history", ""),
            },
            "reasoning_trace": raw_output,
            "prompt_version": PROMPT_VERSION,
        }

        prov = Provenance(
            file=str(MIETIC_PARQUET.relative_to(MIETIC_PARQUET.parents[2])).replace("\\", "/"),
            row_idx=idx,
        )

        rec = build_record(
            record_id=f"mietic_{idx:06d}",
            source="mietic",
            doc_type="case",
            raw_text=raw_input,
            license=LICENSE,
            citation=CITATION,
            provenance=prov,
            case=case,
        )
        records.append(rec)

    n = write_jsonl(OUT_PATH, records)
    log.info("wrote %d records -> %s", n, OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
