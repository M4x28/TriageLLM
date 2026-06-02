"""Step 4 — strategic oversampling of step 3 train.jsonl.

Pipeline:
  1. Load data/rewrite/train.jsonl (29447 records).
  2. Length filter assistant content > 4000 chars (drop ~33 outliers).
  3. Stratified split 95/5 train_pool / val (by doc_type, style, source).
  4. Oversample train_pool:
       - pediatric guideline (who_imci, who_etat) × 10
       - adult guideline    (sats)              × 5
       - mietic cases                            × 1 (pass-through)
  5. Load identity_seed.jsonl (if present): same split, train portion × 10.
  6. Concat + deterministic shuffle.
  7. Write data/augment/train.jsonl + validation.jsonl + augment_stats.json.

Usage:
  python 4_data_augment/augment.py [--skip-identity]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    AUGMENT_DIR,
    REWRITE_DIR,
    length_filter,
    load_jsonl,
    oversample,
    prepend_triage_label,
    setup_logging,
    shuffle_deterministic,
    stratify_split,
    write_jsonl,
)

PEDIATRIC_SOURCES = {"who_imci", "who_etat"}
ADULT_GUIDELINE_SOURCES = {"sats"}
MIETIC_SOURCES = {"mietic"}

OVERSAMPLE_FACTORS = {
    "pediatric_guideline": 10,
    "adult_guideline": 5,
    "mietic_cases": 1,
    "identity": 10,
}


def _classify(record: dict) -> str:
    src = record.get("metadata", {}).get("source")
    if src in PEDIATRIC_SOURCES:
        return "pediatric_guideline"
    if src in ADULT_GUIDELINE_SOURCES:
        return "adult_guideline"
    if src in MIETIC_SOURCES:
        return "mietic_cases"
    if src == "identity":
        return "identity"
    return "unknown"


def main() -> int:
    log = setup_logging("triagellm.augment")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REWRITE_DIR / "train.jsonl"))
    ap.add_argument("--identity-seed", default=str(Path(__file__).parent / "identity_seed.jsonl"))
    ap.add_argument("--output-train", default=str(AUGMENT_DIR / "train.jsonl"))
    ap.add_argument("--output-val", default=str(AUGMENT_DIR / "validation.jsonl"))
    ap.add_argument("--stats", default=str(AUGMENT_DIR / "augment_stats.json"))
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-identity", action="store_true",
                    help="Skip identity_seed.jsonl loading (smoke test)")
    args = ap.parse_args()

    AUGMENT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    log.info("loading %s", in_path)
    records = load_jsonl(in_path)
    log.info("loaded %d records", len(records))

    # Phase 2: move the triage label to the first line of labeled answers,
    # before length filtering so the filter sees the final text.
    n_labeled = prepend_triage_label(records)
    log.info("label-first reformat: prepended header to %d/%d records",
             n_labeled, len(records))

    records, dropped = length_filter(records, max_chars=args.max_chars)
    log.info("length filter: dropped %d (> %d chars), kept %d",
             dropped, args.max_chars, len(records))

    train_pool, val_set = stratify_split(records, val_frac=args.val_frac, seed=args.seed)
    log.info("stratified split: train_pool=%d, val=%d", len(train_pool), len(val_set))

    by_class: dict[str, list[dict]] = {
        "pediatric_guideline": [],
        "adult_guideline": [],
        "mietic_cases": [],
    }
    for r in train_pool:
        cls = _classify(r)
        if cls in by_class:
            by_class[cls].append(r)
        else:
            log.warning("unknown class for record %s (source=%s)",
                        r.get("id"), r.get("metadata", {}).get("source"))

    oversampled: list[dict] = []
    slice_stats: dict[str, dict] = {}
    for cls, group in by_class.items():
        factor = OVERSAMPLE_FACTORS[cls]
        out = oversample(group, factor)
        slice_stats[cls] = {"original": len(group), "factor": factor, "post": len(out)}
        log.info("%-22s: original=%d × %d = %d", cls, len(group), factor, len(out))
        oversampled.extend(out)

    identity_train: list[dict] = []
    identity_val: list[dict] = []
    seed_path = Path(args.identity_seed)
    if args.skip_identity:
        log.info("skipping identity seed (--skip-identity)")
    elif not seed_path.exists():
        log.warning("identity seed not found at %s — proceeding without it", seed_path)
    else:
        log.info("loading identity seed %s", seed_path)
        id_recs = load_jsonl(seed_path)
        id_recs, id_dropped = length_filter(id_recs, max_chars=args.max_chars)
        log.info("identity: loaded %d (length-dropped %d)", len(id_recs), id_dropped)
        id_train_pool, id_val = stratify_split(id_recs, val_frac=args.val_frac, seed=args.seed)
        id_factor = OVERSAMPLE_FACTORS["identity"]
        identity_train = oversample(id_train_pool, id_factor)
        identity_val = id_val
        slice_stats["identity"] = {
            "original": len(id_train_pool),
            "factor": id_factor,
            "post": len(identity_train),
            "validation": len(id_val),
        }
        log.info("identity: train original=%d × %d = %d, val=%d",
                 len(id_train_pool), id_factor, len(identity_train), len(id_val))

    train_final = shuffle_deterministic(oversampled + identity_train, seed=args.seed)
    val_final = shuffle_deterministic(val_set + identity_val, seed=args.seed)

    n_train = write_jsonl(Path(args.output_train), train_final)
    n_val = write_jsonl(Path(args.output_val), val_final)

    stats = {
        "input_records": len(records) + dropped,
        "length_dropped": dropped,
        "post_length_filter": len(records),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "train_total": n_train,
        "validation_total": n_val,
        "slices": slice_stats,
    }
    Path(args.stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("wrote train=%d val=%d stats=%s", n_train, n_val, args.stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
