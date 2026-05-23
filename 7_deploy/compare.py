"""Compare BF16 (step 6) vs Q4_K_M (step 7) evaluation outputs.

Loads both JSON files, computes per-metric deltas, applies the deploy
gate, and writes `data/deploy/compare_bf16_q4km.json` with a decision
field that the user reads before publishing the GGUF to HF Hub.

Usage:
  python 7_deploy/compare.py \
      --bf16 data/eval/internal/qwen3-1.7b.json \
      --q4km data/deploy/eval_q4km/qwen3-1.7b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import COMPARE_FILE, setup_logging


# Deploy gates (see docs/7_deploy.md). All must pass to proceed to publish.
NLL_MAX_DROP = 0.10
ESI_MIN = 0.90
PEDIATRIC_MIN = 0.65
REFUSAL_MAX_DELTA = 0.05


def _delta(q4: float, bf: float) -> float:
    return round(q4 - bf, 4)


def _gate_pass(metrics: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metrics["validation"]["nll_delta"] > NLL_MAX_DROP:
        failures.append(
            f"nll_delta={metrics['validation']['nll_delta']} > {NLL_MAX_DROP}"
        )
    if metrics["triage"]["esi"]["q4km"] < ESI_MIN:
        failures.append(
            f"esi_q4km={metrics['triage']['esi']['q4km']} < {ESI_MIN}"
        )
    if metrics["triage"]["pediatric"]["q4km"] < PEDIATRIC_MIN:
        failures.append(
            f"pediatric_q4km={metrics['triage']['pediatric']['q4km']} < {PEDIATRIC_MIN}"
        )
    if metrics["identity"]["refusal_delta"] > REFUSAL_MAX_DELTA:
        failures.append(
            f"refusal_delta={metrics['identity']['refusal_delta']} > {REFUSAL_MAX_DELTA}"
        )
    return (len(failures) == 0, failures)


def build_comparison(bf16: dict, q4km: dict) -> dict:
    """Build the side-by-side metric comparison dict."""
    val_bf = bf16["validation"]
    val_q4 = q4km["validation"]
    id_bf = bf16["identity"]
    id_q4 = q4km["identity"]
    tri_bf = bf16["triage_and_latency"]
    tri_q4 = q4km["triage_and_latency"]

    cmp = {
        "model": bf16.get("model"),
        "validation": {
            "bf16_nll": val_bf["nll"],
            "q4km_nll": val_q4["nll"],
            "nll_delta": _delta(val_q4["nll"], val_bf["nll"]),
            "bf16_ppl": val_bf["perplexity"],
            "q4km_ppl": val_q4["perplexity"],
            "ppl_delta": _delta(val_q4["perplexity"], val_bf["perplexity"]),
        },
        "identity": {
            "bf16_refusal_rate": id_bf["refusal_rate"],
            "q4km_refusal_rate": id_q4["refusal_rate"],
            "refusal_delta": _delta(id_q4["refusal_rate"], id_bf["refusal_rate"]),
        },
        "triage": {
            "esi": {
                "bf16": tri_bf["esi"]["accuracy"],
                "q4km": tri_q4["esi"]["accuracy"],
                "delta": _delta(tri_q4["esi"]["accuracy"],
                                tri_bf["esi"]["accuracy"]),
            },
            "sats": {
                "bf16": tri_bf["sats"]["accuracy"],
                "q4km": tri_q4["sats"]["accuracy"],
                "delta": _delta(tri_q4["sats"]["accuracy"],
                                tri_bf["sats"]["accuracy"]),
            },
            "pediatric": {
                "bf16": tri_bf["pediatric"]["recall"],
                "q4km": tri_q4["pediatric"]["recall"],
                "delta": _delta(tri_q4["pediatric"]["recall"],
                                tri_bf["pediatric"]["recall"]),
            },
        },
        "latency": {
            "bf16_tokens_per_s_gpu": tri_bf["latency"]["tokens_per_s"],
            "q4km_tokens_per_s_cpu": tri_q4["latency"]["tokens_per_s"],
            "note": "BF16 measured on L40S GPU, Q4_K_M on server CPU "
                    "(mobile ARM proxy). Direct ratio not meaningful.",
        },
    }
    gate_pass, failures = _gate_pass(cmp)
    cmp["gate"] = {
        "thresholds": {
            "nll_max_drop": NLL_MAX_DROP,
            "esi_min": ESI_MIN,
            "pediatric_min": PEDIATRIC_MIN,
            "refusal_max_delta": REFUSAL_MAX_DELTA,
        },
        "pass": gate_pass,
        "failures": failures,
    }
    cmp["decision"] = "publish" if gate_pass else "fallback_q5km"
    return cmp


def main() -> int:
    log = setup_logging("triagellm.deploy.compare")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True,
                    help="path to the BF16 eval JSON (step 6 internal output)")
    ap.add_argument("--q4km", required=True,
                    help="path to the Q4_K_M eval JSON (step 7 eval_gguf output)")
    ap.add_argument("--out", default=str(COMPARE_FILE))
    args = ap.parse_args()

    bf16 = json.loads(Path(args.bf16).read_text(encoding="utf-8"))
    q4km = json.loads(Path(args.q4km).read_text(encoding="utf-8"))

    cmp = build_comparison(bf16, q4km)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cmp, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("wrote %s", out_path)
    log.info("decision: %s", cmp["decision"])
    if not cmp["gate"]["pass"]:
        log.warning("gate FAILED: %s", cmp["gate"]["failures"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
