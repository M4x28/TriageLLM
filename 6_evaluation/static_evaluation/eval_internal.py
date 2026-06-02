"""Internal evaluation on the step-4 validation set.

Computes per-model metrics:
  * validation NLL / perplexity
  * triage ESI accuracy  (parse "ESI N" against gold)
  * triage SATS accuracy (parse "SATS <color>" against gold)
  * pediatric recall     (correct label on records with source in IMCI / ETAT)
  * refusal rate         (10 identity probes)
  * latency tok/s        (GPU bf16 generation, max 200 tokens)
  * style F1 per stile   (caregiver / clinician / field_worker / direct /
                          protocol / red_flag)

Usage:
  CUDA_VISIBLE_DEVICES=0 python 6_evaluation/eval_internal.py --model qwen3-1.7b
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    AUGMENT_DIR,
    INTERNAL_DIR,
    MODELS,
    generate,
    load_jsonl,
    load_merged_model,
    setup_logging,
)
from prompts import (
    IDENTITY_PROBES,
    PEDIATRIC_CONTENT_KEYWORDS,
    SYSTEM_PROMPT,
    extract_triage_labels,
    is_refusal,
)


# Records contribute to the triage F1 only if the gold assistant contains
# an explicit ESI or SATS label. Sections (guidelines) do not, so they are
# evaluated only for pediatric recall and PPL.
def _has_triage_label(text: str) -> bool:
    gold_esi, gold_sats = extract_triage_labels(text)
    return gold_esi is not None or gold_sats is not None


def _compute_validation_nll(model, tokenizer, records: list[dict],
                            n_samples: int) -> tuple[float, float, int]:
    """Token-level mean NLL over a deterministic sub-sample of validation."""
    import torch

    sample = records[:n_samples]
    total_loss = 0.0
    total_tokens = 0
    for rec in sample:
        prompt = tokenizer.apply_chat_template(rec["messages"], tokenize=False)
        ids = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        try:
            with torch.no_grad():
                out = model(**ids, labels=ids["input_ids"])
            loss_val = float(out.loss.item())
        except Exception:
            # Large-vocab models (Qwen3.6) trigger CUDA assert in cross_entropy
            # when running NLL with labels on multi-GPU device_map. Skip record.
            continue
        if not math.isfinite(loss_val):
            continue
        n_tok = ids["input_ids"].numel()
        total_loss += loss_val * n_tok
        total_tokens += n_tok
    avg_nll = total_loss / max(total_tokens, 1) if total_tokens > 0 else float("nan")
    ppl = math.exp(avg_nll) if math.isfinite(avg_nll) and avg_nll < 50 else float("inf")
    return avg_nll, ppl, total_tokens


def _eval_identity(model, tokenizer, log) -> tuple[int, float, list[dict]]:
    """Run the 10 identity probes and tally refusals."""
    log.info("running %d identity probes", len(IDENTITY_PROBES))
    n_refusals = 0
    results: list[dict] = []
    for probe in IDENTITY_PROBES:
        text, dt = generate(model, tokenizer, SYSTEM_PROMPT, probe["prompt"],
                            max_new_tokens=200)
        refused = is_refusal(text)
        if refused:
            n_refusals += 1
        results.append({
            "persona": probe["persona"],
            "prompt": probe["prompt"],
            "response_preview": text[:300],
            "refused": refused,
            "latency_s": round(dt, 2),
        })
    refusal_rate = n_refusals / max(len(IDENTITY_PROBES), 1)
    return n_refusals, refusal_rate, results


def _eval_triage_and_latency(model, tokenizer, records: list[dict],
                             n_samples: int, log,
                             max_new_tokens: int = 1024) -> dict:
    """Triage F1 (ESI + SATS), pediatric content recall, latency, style breakdown.

    Two disjoint subsets are scored:
      * triage F1: records with an explicit triage label in the gold answer
        (MIETIC cases). Compares parsed ESI / SATS labels.
      * pediatric recall: records sourced from WHO IMCI / WHO ETAT, which
        are guideline content (no triage label in gold). Uses keyword
        overlap with PEDIATRIC_CONTENT_KEYWORDS to count how many gold
        keywords the model reproduces; recall = mean overlap fraction.
    """
    triage_sample = [r for r in records[:n_samples]
                     if _has_triage_label(r["messages"][2]["content"])]
    pediatric_sample = [r for r in records[:n_samples]
                        if r.get("metadata", {}).get("source") in ("who_imci", "who_etat")]
    log.info("triage F1 on %d labeled records, pediatric content recall "
             "on %d IMCI/ETAT records", len(triage_sample), len(pediatric_sample))

    correct_esi = 0
    correct_sats = 0
    n_esi = n_sats = 0
    # Phase 2 format compliance counters: how many responses contain a
    # parseable triage label, regardless of whether it matches gold.
    esi_present = 0
    sats_present = 0
    either_present = 0
    latencies_s: list[float] = []
    tokens_emitted: list[int] = []
    per_style: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "esi_ok": 0, "sats_ok": 0}
    )

    for rec in triage_sample:
        messages = rec["messages"]
        gold_assistant = messages[2]["content"]
        gold_esi, gold_sats = extract_triage_labels(gold_assistant)

        text, dt = generate(model, tokenizer,
                            messages[0]["content"], messages[1]["content"],
                            max_new_tokens=max_new_tokens)
        pred_esi, pred_sats = extract_triage_labels(text)

        # Format compliance: was a triage label emitted at all?
        if pred_esi is not None:
            esi_present += 1
        if pred_sats is not None:
            sats_present += 1
        if pred_esi is not None or pred_sats is not None:
            either_present += 1

        out_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        latencies_s.append(dt)
        tokens_emitted.append(max(out_tokens, 1))

        meta = rec.get("metadata", {})
        style = meta.get("style", "?")
        per_style[style]["n"] += 1

        if gold_esi is not None:
            n_esi += 1
            if pred_esi == gold_esi:
                correct_esi += 1
                per_style[style]["esi_ok"] += 1
        if gold_sats is not None:
            n_sats += 1
            if pred_sats == gold_sats:
                correct_sats += 1
                per_style[style]["sats_ok"] += 1

    pediatric_overlaps: list[float] = []
    for rec in pediatric_sample:
        messages = rec["messages"]
        gold_text = messages[2]["content"].lower()
        text, dt = generate(model, tokenizer,
                            messages[0]["content"], messages[1]["content"],
                            max_new_tokens=max_new_tokens)
        latencies_s.append(dt)
        tokens_emitted.append(
            max(len(tokenizer(text, add_special_tokens=False)["input_ids"]), 1)
        )
        # Keyword overlap: how many critical pediatric tokens that appear in
        # the gold answer are reproduced by the model response.
        gold_hits = [kw for kw in PEDIATRIC_CONTENT_KEYWORDS if kw in gold_text]
        if not gold_hits:
            continue
        model_lower = text.lower()
        matched = sum(1 for kw in gold_hits if kw in model_lower)
        pediatric_overlaps.append(matched / len(gold_hits))
    pediatric_recall = (
        sum(pediatric_overlaps) / len(pediatric_overlaps)
        if pediatric_overlaps else 0.0
    )
    pediatric_total = len(pediatric_overlaps)

    # Latency in tokens/sec: aggregate tokens emitted / aggregate seconds.
    total_tokens = sum(tokens_emitted)
    total_seconds = sum(latencies_s) or 1e-6
    tok_s = total_tokens / total_seconds

    def acc(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    n_triage = len(triage_sample)
    return {
        "n_labeled_samples": n_triage,
        "esi": {"n": n_esi, "correct": correct_esi, "accuracy": acc(correct_esi, n_esi)},
        "sats": {"n": n_sats, "correct": correct_sats, "accuracy": acc(correct_sats, n_sats)},
        "format": {
            "esi_present": esi_present,
            "sats_present": sats_present,
            "either_present": either_present,
            "esi_present_rate": acc(esi_present, n_triage),
            "sats_present_rate": acc(sats_present, n_triage),
            "either_present_rate": acc(either_present, n_triage),
        },
        "pediatric": {
            "n": pediatric_total,
            "metric": "gold_keyword_overlap",
            "recall": round(pediatric_recall, 4),
        },
        "latency": {
            "mean_s_per_gen": round(total_seconds / max(len(latencies_s), 1), 3),
            "tokens_per_s": round(tok_s, 2),
            "p95_s_per_gen": round(
                sorted(latencies_s)[int(0.95 * max(len(latencies_s) - 1, 0))]
                if latencies_s else 0.0,
                3,
            ),
        },
        "by_style": {
            style: {
                "n": d["n"],
                "esi_accuracy": acc(d["esi_ok"], d["n"]),
                "sats_accuracy": acc(d["sats_ok"], d["n"]),
            }
            for style, d in per_style.items()
        },
    }


def main() -> int:
    log = setup_logging("triagellm.eval.internal")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--validation-jsonl",
                    default=str(AUGMENT_DIR / "validation.jsonl"))
    ap.add_argument("--n-ppl-samples", type=int, default=200,
                    help="records used for NLL / PPL computation")
    ap.add_argument("--n-triage-samples", type=int, default=400,
                    help="records used for triage F1 + latency (filtered to "
                         "those with explicit gold labels)")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="cap per-record generation length")
    ap.add_argument("--load-4bit", action="store_true",
                    help="load merged checkpoint in BnB NF4 4-bit (single GPU, faster eval)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERNAL_DIR / f"{spec.slug}.json"
    log.info("eval_internal model=%s -> %s", spec.slug, out_path)

    log.info("loading merged checkpoint (4bit=%s)", args.load_4bit)
    tokenizer, model = load_merged_model(spec, use_4bit=args.load_4bit)

    log.info("loading validation JSONL")
    records = load_jsonl(Path(args.validation_jsonl))

    nll, ppl, n_tokens = _compute_validation_nll(
        model, tokenizer, records, args.n_ppl_samples,
    )
    log.info("NLL=%.4f PPL=%.2f over %d tokens", nll, ppl, n_tokens)

    n_refusals, refusal_rate, identity_results = _eval_identity(model, tokenizer, log)
    log.info("refusal_rate=%.1f%% (%d/%d)",
             refusal_rate * 100, n_refusals, len(IDENTITY_PROBES))

    triage_metrics = _eval_triage_and_latency(
        model, tokenizer, records, args.n_triage_samples, log,
        max_new_tokens=args.max_new_tokens,
    )
    log.info("ESI acc=%.4f SATS acc=%.4f pediatric recall=%.4f tok/s=%.2f "
             "fmt either_present=%.4f",
             triage_metrics["esi"]["accuracy"],
             triage_metrics["sats"]["accuracy"],
             triage_metrics["pediatric"]["recall"],
             triage_metrics["latency"]["tokens_per_s"],
             triage_metrics["format"]["either_present_rate"])

    summary = {
        "model": spec.slug,
        "hf_or_path": str(spec.hf_id),
        "validation": {
            "n_samples": min(args.n_ppl_samples, len(records)),
            "tokens": n_tokens,
            "nll": round(nll, 4),
            "perplexity": round(ppl, 2),
        },
        "identity": {
            "n_probes": len(IDENTITY_PROBES),
            "n_refusals": n_refusals,
            "refusal_rate": round(refusal_rate, 4),
            "results": identity_results,
        },
        "triage_and_latency": triage_metrics,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
