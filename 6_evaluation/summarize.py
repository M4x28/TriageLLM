"""Aggregate per-model JSON reports into a single ranking + summary.

Composite weights are fixed in `COMPOSITE_WEIGHTS` and reflect the deploy
goals confirmed by the user:
  * primary objective: help triage decisions
  * secondary objective: cut triage time (latency)

Disqualifying hardware gates are applied first; surviving models are
ranked by composite score and written to data/eval/summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    EVAL_DIR,
    EXTERNAL_DIR,
    INTERNAL_DIR,
    MODELS,
    REFERENCE_MODEL_SLUG,
    setup_logging,
)


# Composite weights (must sum to 1.0). Each component is normalized to [0, 1]
# with "higher is better" before weighting.
COMPOSITE_WEIGHTS = {
    "esi_accuracy":       0.25,
    "pediatric_recall":   0.20,
    "non_refusal_rate":   0.20,
    "latency_score":      0.20,
    "sats_accuracy":      0.15,
}

# Throughput saturation point for the latency component (tok/s on L40S bf16).
# Above this value `latency_score` caps at 1.0.
LATENCY_TARGET_TOK_S = 50.0

# Disqualifying hardware gates: a model is excluded from ranking if any of
# these thresholds is violated.
GATES = {
    "min_latency_tok_s": 10.0,
    "max_refusal_rate":  0.30,
    "min_pediatric_recall": 0.40,
}


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latency_score(tok_s: float) -> float:
    """Map throughput to [0, 1] saturating at LATENCY_TARGET_TOK_S."""
    if tok_s <= 0:
        return 0.0
    return min(tok_s / LATENCY_TARGET_TOK_S, 1.0)


def _per_model_view(slug: str, internal: dict, external: dict | None) -> dict:
    """Flatten the metrics needed for ranking into a single dict."""
    tl = internal["triage_and_latency"]
    esi_acc = tl["esi"]["accuracy"]
    sats_acc = tl["sats"]["accuracy"]
    pediatric = tl["pediatric"]["recall"]
    tok_s = tl["latency"]["tokens_per_s"]
    refusal_rate = internal["identity"]["refusal_rate"]

    view: dict = {
        "model": slug,
        "esi_accuracy": esi_acc,
        "sats_accuracy": sats_acc,
        "pediatric_recall": pediatric,
        "refusal_rate": refusal_rate,
        "non_refusal_rate": round(1.0 - refusal_rate, 4),
        "tokens_per_s_gpu": tok_s,
        "latency_score": round(_latency_score(tok_s), 4),
        "validation_perplexity": internal["validation"]["perplexity"],
    }
    if external is not None:
        view["external"] = {
            "medqa_accuracy":   external.get("medqa", {}).get("accuracy"),
            "pubmedqa_accuracy": external.get("pubmedqa", {}).get("accuracy"),
            "mmlu_clinical_accuracy": external.get("mmlu_clinical", {}).get("accuracy"),
        }
    return view


def _composite(view: dict) -> float:
    components = {
        "esi_accuracy":     view["esi_accuracy"],
        "pediatric_recall": view["pediatric_recall"],
        "non_refusal_rate": view["non_refusal_rate"],
        "latency_score":    view["latency_score"],
        "sats_accuracy":    view["sats_accuracy"],
    }
    return round(sum(COMPOSITE_WEIGHTS[k] * v for k, v in components.items()), 4)


def _check_gates(view: dict) -> tuple[bool, list[str]]:
    """Return (passes, reasons_for_failure)."""
    fails: list[str] = []
    if view["tokens_per_s_gpu"] < GATES["min_latency_tok_s"]:
        fails.append(
            f"latency_tok_s_gpu {view['tokens_per_s_gpu']} "
            f"< gate {GATES['min_latency_tok_s']}"
        )
    if view["refusal_rate"] > GATES["max_refusal_rate"]:
        fails.append(
            f"refusal_rate {view['refusal_rate']} "
            f"> gate {GATES['max_refusal_rate']}"
        )
    if view["pediatric_recall"] < GATES["min_pediatric_recall"]:
        fails.append(
            f"pediatric_recall {view['pediatric_recall']} "
            f"< gate {GATES['min_pediatric_recall']}"
        )
    return (not fails, fails)


def main() -> int:
    log = setup_logging("triagellm.eval.summarize")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EVAL_DIR / "summary.json"))
    args = ap.parse_args()

    # Collect the 3 SFT candidates.
    candidates: list[dict] = []
    for slug in MODELS:
        internal = _read(INTERNAL_DIR / f"{slug}.json")
        if internal is None:
            log.warning("missing internal report for %s, skipping", slug)
            continue
        external = _read(EXTERNAL_DIR / f"{slug}.json")
        view = _per_model_view(slug, internal, external)
        passes, fail_reasons = _check_gates(view)
        view["gates_passed"] = passes
        view["gate_failures"] = fail_reasons
        view["composite_score"] = _composite(view)
        candidates.append(view)

    # Reference model is reported separately, never ranked.
    reference: dict | None = None
    ref_external = _read(EXTERNAL_DIR / f"{REFERENCE_MODEL_SLUG}.json")
    if ref_external is not None:
        reference = {
            "model": REFERENCE_MODEL_SLUG,
            "is_reference": True,
            "external": {
                "medqa_accuracy":         ref_external.get("medqa", {}).get("accuracy"),
                "pubmedqa_accuracy":      ref_external.get("pubmedqa", {}).get("accuracy"),
                "mmlu_clinical_accuracy": ref_external.get("mmlu_clinical", {}).get("accuracy"),
            },
        }

    # Rank only models that pass the gates; failures listed separately.
    ranked = sorted(
        [c for c in candidates if c["gates_passed"]],
        key=lambda c: c["composite_score"], reverse=True,
    )
    rejected = [c for c in candidates if not c["gates_passed"]]

    summary = {
        "weights": COMPOSITE_WEIGHTS,
        "latency_target_tok_s": LATENCY_TARGET_TOK_S,
        "gates": GATES,
        "ranking": ranked,
        "disqualified": rejected,
        "reference": reference,
        "decision": {
            "primary": ranked[0]["model"] if ranked else None,
            "gap_to_second": (
                round(ranked[0]["composite_score"]
                      - ranked[1]["composite_score"], 4)
                if len(ranked) >= 2 else None
            ),
        },
    }
    Path(args.out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    log.info("wrote %s", args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
