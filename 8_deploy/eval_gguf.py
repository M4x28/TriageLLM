"""Post-quantization evaluation on the step-4 validation set.

Mirrors `6_evaluation/eval_internal.py` so the output JSON has the same
schema and can be diffed against the BF16 baseline. Differences:

  * Backend is `llama-cpp-python` reading a GGUF Q4_K_M file (CPU by
    default to approximate the Android ARM Neon path).
  * The HF tokenizer of the merged checkpoint is still loaded but only
    to apply the chat template uniformly. All actual inference goes
    through `llama_cpp.Llama`.

Usage:
  python 7_deploy/eval_gguf.py --model qwen3-1.7b \
      --gguf data/deploy/gguf/qwen3-1.7b-q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from common import (
    AUGMENT_DIR,
    CHECKPOINT_DIR,
    EVAL_Q4KM_DIR,
    IDENTITY_PROBES,
    MODELS,
    PEDIATRIC_CONTENT_KEYWORDS,
    SYSTEM_PROMPT,
    extract_triage_labels,
    generate_gguf,
    is_refusal,
    load_gguf_model,
    load_jsonl,
    setup_logging,
)


def _has_triage_label(text: str) -> bool:
    """Same filter as step 6: keep records whose gold answer carries an
    explicit ESI or SATS label."""
    gold_esi, gold_sats = extract_triage_labels(text)
    return gold_esi is not None or gold_sats is not None


def _load_hf_tokenizer(slug: str):
    """Load the HF tokenizer of the merged checkpoint for chat-template
    application only. No model is loaded here."""
    from transformers import AutoTokenizer

    source = CHECKPOINT_DIR / slug / "merged"
    return AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)


def _apply_template(tokenizer, messages: list[dict]) -> str:
    """Apply the chat template embedded in the HF tokenizer."""
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False)


def _compute_validation_nll(llama, hf_tokenizer, records: list[dict],
                            n_samples: int, log) -> tuple[float, float, int]:
    """Token-level mean NLL via llama.cpp `logits_all` evaluation.

    Replicates the step-6 logic (NLL over the full chat-template-wrapped
    record) so the BF16 vs Q4_K_M comparison is apples-to-apples.
    """
    sample = records[:n_samples]
    total_nll_sum = 0.0
    total_tokens = 0
    log.info("computing NLL on %d records", len(sample))
    for i, rec in enumerate(sample):
        prompt = _apply_template(hf_tokenizer, rec["messages"])
        # llama-cpp returns per-token logprobs via echo=True.
        out = llama(prompt, max_tokens=0, logprobs=0, echo=True)
        token_logprobs = out["choices"][0]["logprobs"]["token_logprobs"]
        valid = [lp for lp in token_logprobs if lp is not None]
        if not valid:
            continue
        total_nll_sum += -sum(valid)
        total_tokens += len(valid)
        if (i + 1) % 25 == 0:
            log.info("  NLL progress %d/%d", i + 1, len(sample))
    avg_nll = total_nll_sum / max(total_tokens, 1)
    ppl = math.exp(avg_nll) if avg_nll < 50 else float("inf")
    return avg_nll, ppl, total_tokens


def _eval_identity(llama, log,
                   max_new_tokens: int = 200) -> tuple[int, float, list[dict]]:
    """Run the 10 identity probes and tally refusals."""
    log.info("running %d identity probes", len(IDENTITY_PROBES))
    n_refusals = 0
    results: list[dict] = []
    for probe in IDENTITY_PROBES:
        text, dt, _ = generate_gguf(llama, SYSTEM_PROMPT, probe["prompt"],
                                    max_new_tokens=max_new_tokens)
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


def _eval_triage_and_latency(llama, records: list[dict], n_samples: int, log,
                             max_new_tokens: int = 1024) -> dict:
    """Triage F1 + pediatric keyword overlap + CPU latency + style F1.

    Same metric definitions as step 6, only the inference backend changes.
    """
    triage_sample = [r for r in records[:n_samples]
                     if _has_triage_label(r["messages"][2]["content"])]
    pediatric_sample = [r for r in records[:n_samples]
                        if r.get("metadata", {}).get("source") in ("who_imci", "who_etat")]
    log.info("triage on %d labeled records, pediatric content recall on %d "
             "IMCI/ETAT records", len(triage_sample), len(pediatric_sample))

    correct_esi = 0
    correct_sats = 0
    n_esi = n_sats = 0
    latencies_s: list[float] = []
    tokens_emitted: list[int] = []
    per_style: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "esi_ok": 0, "sats_ok": 0}
    )

    for rec in triage_sample:
        messages = rec["messages"]
        gold_assistant = messages[2]["content"]
        gold_esi, gold_sats = extract_triage_labels(gold_assistant)

        text, dt, n_tok = generate_gguf(
            llama, messages[0]["content"], messages[1]["content"],
            max_new_tokens=max_new_tokens,
        )
        pred_esi, pred_sats = extract_triage_labels(text)
        latencies_s.append(dt)
        tokens_emitted.append(n_tok)

        style = rec.get("metadata", {}).get("style", "?")
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
        text, dt, n_tok = generate_gguf(
            llama, messages[0]["content"], messages[1]["content"],
            max_new_tokens=max_new_tokens,
        )
        latencies_s.append(dt)
        tokens_emitted.append(n_tok)
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

    total_tokens = sum(tokens_emitted)
    total_seconds = sum(latencies_s) or 1e-6
    tok_s = total_tokens / total_seconds

    def acc(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    return {
        "n_labeled_samples": len(triage_sample),
        "esi": {"n": n_esi, "correct": correct_esi,
                "accuracy": acc(correct_esi, n_esi)},
        "sats": {"n": n_sats, "correct": correct_sats,
                 "accuracy": acc(correct_sats, n_sats)},
        "pediatric": {
            "n": len(pediatric_overlaps),
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
    log = setup_logging("triagellm.deploy.eval_gguf")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gguf", required=True,
                    help="path to the GGUF file to evaluate")
    ap.add_argument("--validation-jsonl",
                    default=str(AUGMENT_DIR / "validation.jsonl"))
    ap.add_argument("--n-ppl-samples", type=int, default=200)
    ap.add_argument("--n-triage-samples", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=None,
                    help="llama.cpp CPU threads (default: auto)")
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="GPU layer offload count (0 = CPU only, mobile proxy)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    EVAL_Q4KM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_Q4KM_DIR / f"{spec.slug}.json"
    log.info("eval_gguf model=%s gguf=%s -> %s", spec.slug, args.gguf, out_path)

    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        log.error("GGUF file not found: %s", gguf_path)
        return 1

    log.info("loading HF tokenizer for chat template")
    hf_tokenizer = _load_hf_tokenizer(spec.slug)

    log.info("loading validation JSONL")
    records = load_jsonl(Path(args.validation_jsonl))

    # Phase 1: load GGUF with logits_all=True for NLL only (slow path).
    if args.n_ppl_samples > 0:
        log.info("phase 1 (NLL): loading GGUF with logits_all=True")
        t0 = time.time()
        llama_nll = load_gguf_model(gguf_path, n_ctx=args.n_ctx,
                                    n_threads=args.n_threads,
                                    n_gpu_layers=args.n_gpu_layers,
                                    logits_all=True)
        log.info("GGUF (NLL mode) loaded in %.1fs", time.time() - t0)
        nll, ppl, n_tokens = _compute_validation_nll(
            llama_nll, hf_tokenizer, records, args.n_ppl_samples, log,
        )
        log.info("NLL=%.4f PPL=%.2f over %d tokens", nll, ppl, n_tokens)
        del llama_nll
        import gc; gc.collect()
    else:
        log.info("skipping NLL phase (n_ppl_samples=0)")
        nll, ppl, n_tokens = 0.0, 0.0, 0

    # Phase 2: reload with logits_all=False for fast generation tasks.
    log.info("phase 2 (generate): loading GGUF with logits_all=False")
    t0 = time.time()
    llama = load_gguf_model(gguf_path, n_ctx=args.n_ctx,
                            n_threads=args.n_threads,
                            n_gpu_layers=args.n_gpu_layers,
                            logits_all=False)
    log.info("GGUF (gen mode) loaded in %.1fs", time.time() - t0)

    n_refusals, refusal_rate, identity_results = _eval_identity(llama, log)
    log.info("refusal_rate=%.1f%% (%d/%d)",
             refusal_rate * 100, n_refusals, len(IDENTITY_PROBES))

    triage_metrics = _eval_triage_and_latency(
        llama, records, args.n_triage_samples, log,
        max_new_tokens=args.max_new_tokens,
    )
    log.info("ESI acc=%.4f SATS acc=%.4f pediatric recall=%.4f tok/s=%.2f",
             triage_metrics["esi"]["accuracy"],
             triage_metrics["sats"]["accuracy"],
             triage_metrics["pediatric"]["recall"],
             triage_metrics["latency"]["tokens_per_s"])

    summary = {
        "model": spec.slug,
        "backend": "llama-cpp-python",
        "gguf_path": str(gguf_path),
        "gguf_quant": "Q4_K_M",
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
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                   default=float),
                        encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
