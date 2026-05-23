"""Clean PDF → <source>_sections.jsonl via heading-based split + sliding-window fallback.

Usage:
  python clean_pdf.py --source sats|imci|etat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    MinHashDedup,
    PROCESSED_DIR,
    Provenance,
    RAW_DIR,
    breadcrumb_for_page,
    build_record,
    extract_pdf_outline,
    extract_pdf_pages,
    is_junk_section,
    length_filter,
    setup_logging,
    sliding_window,
    split_into_sections,
    write_jsonl,
)


SOURCES = {
    "sats": {
        "pdf": "pdf/sats_manual.pdf",
        "out": "sats_sections.jsonl",
        "source_key": "sats",
        "citation": "SATS Training Manual 2012 (EMSSA)",
    },
    "imci": {
        "pdf": "pdf/who_imci.pdf",
        "out": "imci_sections.jsonl",
        "source_key": "who_imci",
        "citation": "WHO IMCI Chart Booklet",
    },
    "etat": {
        "pdf": "pdf/who_etat.pdf",
        "out": "etat_sections.jsonl",
        "source_key": "who_etat",
        "citation": "WHO ETAT Participant Manual",
    },
}

MANIFEST_PATH = RAW_DIR / "manifest.json"


def _read_license(source_key: str, rel_path: str) -> str:
    if not MANIFEST_PATH.exists():
        return "UNKNOWN"
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    key = f"{source_key}::{rel_path}"
    entry = data.get(key)
    return entry["license"] if entry else "UNKNOWN"


def main() -> int:
    log = setup_logging("triagellm.clean.pdf")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--window", type=int, default=1500)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--max-section", type=int, default=3000,
                    help="Sections above this size are sliding-window split")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=5000)
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    pdf_path = RAW_DIR / cfg["pdf"]
    if not pdf_path.exists():
        log.error("PDF missing: %s — run step 1 first", pdf_path)
        return 1

    log.info("extracting text from %s", pdf_path)
    pages = extract_pdf_pages(pdf_path)
    log.info("%d pages extracted", len(pages))

    # build char-offset per page boundary
    page_offsets = [0]
    full_text_parts = []
    for p in pages:
        full_text_parts.append(p)
        page_offsets.append(page_offsets[-1] + len(p) + 1)
        full_text_parts.append("\n")  # page separator
    full_text = "\n".join(pages)
    # recompute offsets aligned to the joined string
    page_offsets = [0]
    for p in pages[:-1]:
        page_offsets.append(page_offsets[-1] + len(p) + 1)

    outline = extract_pdf_outline(pdf_path)
    log.info("outline entries: %d", len(outline))

    sections = split_into_sections(full_text, page_offsets)
    log.info("heading-based sections: %d", len(sections))

    rel_path = cfg["pdf"]
    license_str = _read_license(cfg["source_key"], rel_path)
    dedup = MinHashDedup(threshold=0.85)
    records = []
    skipped_length = 0
    skipped_dup = 0
    skipped_junk = 0

    sub_idx = 0
    for sec in sections:
        body = sec["text"]
        page_start = sec["page_start"]
        page_end = sec["page_end"]
        crumb = breadcrumb_for_page(outline, page_start - 1)

        # Drop TOC/acknowledgements/copyright-page sections that match the
        # clinical-header regex by accident (e.g. "Triage" appears in TOC)
        if is_junk_section(body):
            skipped_junk += 1
            continue

        chunks = (
            sliding_window(body, window=args.window, overlap=args.overlap)
            if len(body) > args.max_section
            else [body]
        )

        for chunk in chunks:
            if not length_filter(chunk, args.min_chars, args.max_chars):
                skipped_length += 1
                continue
            if dedup.is_duplicate(chunk):
                skipped_dup += 1
                continue

            section_payload = {
                "section_type": sec["section_type"],
                "header": sec["header"],
                "breadcrumb": crumb,
                "text": chunk,
            }
            prov = Provenance(
                file=f"data/raw/{rel_path}",
                page_start=page_start,
                page_end=page_end,
            )
            rec = build_record(
                record_id=f"{cfg['source_key']}_{sub_idx:04d}",
                source=cfg["source_key"],
                doc_type="guideline_section",
                raw_text=chunk,
                license=license_str,
                citation=cfg["citation"],
                provenance=prov,
                section=section_payload,
            )
            records.append(rec)
            sub_idx += 1

    out_path = PROCESSED_DIR / cfg["out"]
    n = write_jsonl(out_path, records)
    log.info("wrote %d records -> %s (skipped: %d junk, %d length, %d dup)",
             n, out_path, skipped_junk, skipped_length, skipped_dup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
