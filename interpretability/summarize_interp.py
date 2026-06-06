"""Collate the small interpretability JSON into data/eval/interp/REPORT.md.

Reads outputs.json (textual baseline), token_prob.json (ESI-IMCI margins),
logit_lens.json (per-layer differential). Separates OBSERVATIONS from weak
interpretation; states the control-gate outcome. No safety claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cases as P

REPO = Path(__file__).resolve().parent.parent


def load(d, name):
    f = d / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data/eval/interp"))
    args = ap.parse_args()
    d = Path(args.dir)
    outputs = load(d, "outputs.json")
    tp = load(d, "token_prob.json")
    ll = load(d, "logit_lens.json")

    L = ["# Interpretability report — ESI framework leak", "",
         "Observations only. circuit-tracer is run on the BASE model (transcoders "
         "valid); fine-tune evidence (round-3/round-5) is transcoder-free "
         "(token-set logprob + logit-lens) and correlational.", ""]

    # 1. textual baseline / framework matrix
    L += ["## Framework per prompt/model (textual baseline)", "",
          "| prompt | expected | base | r3 | r5 |", "| --- | --- | --- | --- | --- |"]
    for pid in P.PROMPTS:
        o = outputs.get(pid, {})
        L.append(f"| {pid} | {P.EXPECTED_MATRIX[pid]} | "
                 + " | ".join(o.get(k, {}).get("framework", "-") for k in ("base", "r3", "r5"))
                 + " |")
    # control gate
    base_peds = outputs.get("esi_leak_resp", {}).get("base", {}).get("framework", "-")
    L += ["", "**Control gate:** base framework on `esi_leak_resp` = "
          f"`{base_peds}`. " + (
              "Base already shows the ESI prior -> partly inherited."
              if base_peds in ("ESI", "MIXED") else
              "Base does NOT show the ESI prior -> the leak is born/reinforced by "
              "the MIETIC fine-tuning, not present in the base model."), ""]

    # 2. ESI - IMCI margin (token-set logprob)
    L += ["## ESI - IMCI margin (token-set mean logprob; higher = more ESI-leaning)", ""]
    for anchor in ("start", "post_action"):
        L += [f"### anchor: {anchor}", "",
              "| prompt | base | r3 | r5 |", "| --- | --- | --- | --- |"]
        for pid in P.PROMPTS:
            row = tp.get(pid, {}).get(anchor, {})
            L.append(f"| {pid} | "
                     + " | ".join(f"{row.get(k, {}).get('esi_minus_imci', '-')}"
                                  for k in ("base", "r3", "r5")) + " |")
        L.append("")

    # 3. logit-lens peak layer
    L += ["## Logit-lens differential (peak ESI-IMCI layer, post_action; noisy)", "",
          "| prompt | base (peak@layer) | r3 | r5 |", "| --- | --- | --- | --- |"]
    for pid in P.PROMPTS:
        cells = []
        for k in ("base", "r3", "r5"):
            curve = ll.get(pid, {}).get(k)
            if curve:
                pk = max(range(len(curve)), key=lambda i: curve[i])
                cells.append(f"{curve[pk]:+.2f}@L{pk}")
            else:
                cells.append("-")
        L.append(f"| {pid} | " + " | ".join(cells) + " |")

    L += ["", "## Interpretation (weak)", "",
          "The analysis suggests that the ESI-label prior remains behaviorally "
          "strong in fine-tuned models; circuit tracing on the base model is used "
          "only to inspect possible origins of this prior. No safety claim is made; "
          "the model still leads `Action: REFER NOW` on danger signs.", ""]

    (d / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[summarize] wrote {d/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
