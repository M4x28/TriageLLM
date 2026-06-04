"""External benchmark evaluation: MedQA-USMLE + PubMedQA + MMLU-Med.

Downloads each dataset from the HuggingFace Hub (HF token already configured
on the server) and scores the model with greedy generation + lightweight
answer-letter / yes-no-maybe parsing.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 6_evaluation/eval_external.py --model qwen3-1.7b
  CUDA_VISIBLE_DEVICES=0 python 6_evaluation/eval_external.py --reference   # MedGemma
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    EXTERNAL_DIR,
    MODELS,
    REFERENCE_MODEL_SPEC,
    generate,
    load_merged_model,
    setup_logging,
)
from prompts import (
    MEDQA_TEMPLATE,
    PUBMEDQA_TEMPLATE,
    SYSTEM_PROMPT,
    extract_mc_letter,
    extract_yes_no_maybe,
)


def _eval_medqa(model, tokenizer, n: int, log) -> dict:
    """MedQA-USMLE English test subset, 4-choice accuracy."""
    from datasets import load_dataset

    log.info("loading MedQA test split")
    ds = load_dataset(
        "bigbio/med_qa", "med_qa_en_source", split="test",
        trust_remote_code=True,
    )
    sample = ds.select(range(min(n, len(ds))))

    correct = 0
    rows: list[dict] = []
    for ex in sample:
        # Each example has 'options' as list of {'key':'A'..'D','value':'text'}.
        opts = {opt["key"]: opt["value"] for opt in ex["options"]}
        prompt = MEDQA_TEMPLATE.format(
            question=ex["question"],
            a=opts.get("A", ""), b=opts.get("B", ""),
            c=opts.get("C", ""), d=opts.get("D", ""),
        )
        text, _ = generate(model, tokenizer, SYSTEM_PROMPT, prompt,
                           max_new_tokens=64)
        pred = extract_mc_letter(text)
        gold = ex["answer_idx"]
        is_correct = pred == gold
        correct += int(is_correct)
        rows.append({
            "id": ex.get("meta_info", {}).get("question_id") if isinstance(
                ex.get("meta_info"), dict
            ) else None,
            "gold": gold, "pred": pred, "correct": is_correct,
            "response_preview": text[:160],
        })
    n_eval = len(sample)
    acc = correct / max(n_eval, 1)
    log.info("MedQA: %d/%d = %.4f", correct, n_eval, acc)
    return {"n": n_eval, "correct": correct, "accuracy": round(acc, 4),
            "samples": rows[:20]}  # keep first 20 for spot check


def _eval_pubmedqa(model, tokenizer, n: int, log) -> dict:
    """PubMedQA PQA-L test, yes/no/maybe accuracy."""
    from datasets import load_dataset

    log.info("loading PubMedQA pqa_labeled")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train",
                      trust_remote_code=True)
    sample = ds.select(range(min(n, len(ds))))

    correct = 0
    rows: list[dict] = []
    for ex in sample:
        context = " ".join(ex["context"]["contexts"])
        prompt = PUBMEDQA_TEMPLATE.format(context=context[:3000],
                                          question=ex["question"])
        text, _ = generate(model, tokenizer, SYSTEM_PROMPT, prompt,
                           max_new_tokens=64)
        pred = extract_yes_no_maybe(text)
        gold = ex["final_decision"].lower()
        is_correct = pred == gold
        correct += int(is_correct)
        rows.append({
            "id": ex.get("pubid"),
            "gold": gold, "pred": pred, "correct": is_correct,
            "response_preview": text[:160],
        })
    n_eval = len(sample)
    acc = correct / max(n_eval, 1)
    log.info("PubMedQA: %d/%d = %.4f", correct, n_eval, acc)
    return {"n": n_eval, "correct": correct, "accuracy": round(acc, 4),
            "samples": rows[:20]}


def _eval_mmlu_med(model, tokenizer, n: int, log) -> dict:
    """MMLU clinical-medicine subset, 4-choice accuracy."""
    from datasets import load_dataset

    log.info("loading MMLU clinical_knowledge test split")
    ds = load_dataset("cais/mmlu", "clinical_knowledge", split="test",
                      trust_remote_code=True)
    sample = ds.select(range(min(n, len(ds))))

    correct = 0
    rows: list[dict] = []
    for ex in sample:
        prompt = MEDQA_TEMPLATE.format(
            question=ex["question"],
            a=ex["choices"][0], b=ex["choices"][1],
            c=ex["choices"][2], d=ex["choices"][3],
        )
        text, _ = generate(model, tokenizer, SYSTEM_PROMPT, prompt,
                           max_new_tokens=64)
        pred = extract_mc_letter(text)
        gold = "ABCD"[ex["answer"]]
        is_correct = pred == gold
        correct += int(is_correct)
        rows.append({
            "gold": gold, "pred": pred, "correct": is_correct,
            "response_preview": text[:160],
        })
    n_eval = len(sample)
    acc = correct / max(n_eval, 1)
    log.info("MMLU-Med: %d/%d = %.4f", correct, n_eval, acc)
    return {"n": n_eval, "correct": correct, "accuracy": round(acc, 4),
            "samples": rows[:20]}


def main() -> int:
    log = setup_logging("triagellm.eval.external")
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", choices=list(MODELS),
                     help="one of the 3 SFT candidates (merged checkpoint)")
    grp.add_argument("--reference", action="store_true",
                     help="evaluate the reference model MedGemma-4B-it")
    ap.add_argument("--n-medqa", type=int, default=100)
    ap.add_argument("--n-pubmedqa", type=int, default=100)
    ap.add_argument("--n-mmlu", type=int, default=50)
    args = ap.parse_args()

    if args.reference:
        spec = REFERENCE_MODEL_SPEC
    else:
        spec = MODELS[args.model]

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXTERNAL_DIR / f"{spec.slug}.json"
    log.info("eval_external model=%s -> %s", spec.slug, out_path)

    tokenizer, model = load_merged_model(spec, allow_reference=args.reference)

    medqa = _eval_medqa(model, tokenizer, args.n_medqa, log)
    pubmedqa = _eval_pubmedqa(model, tokenizer, args.n_pubmedqa, log)
    mmlu = _eval_mmlu_med(model, tokenizer, args.n_mmlu, log)

    summary = {
        "model": spec.slug,
        "is_reference": args.reference,
        "hf_or_path": str(spec.hf_id),
        "medqa": medqa,
        "pubmedqa": pubmedqa,
        "mmlu_clinical": mmlu,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
