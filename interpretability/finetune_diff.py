"""Transcoder-free fine-tune comparison: base vs round-3 vs round-5.

Two measures, neither needs transcoders (so they are valid on the fine-tuned
weights, unlike attribution graphs):

1. Token-set sequence logprob (length-normalised) of the ESI phrase-set vs the
   IMCI phrase-set, measured at two input-constant anchors:
     - "start"       : the generation prompt (first assistant token);
     - "post_action" : prompt + "Action: REFER NOW\n\n" (the realistic point where
                       the leak actually picks a framework — ESI Level tends to
                       appear AFTER the action line).
   Report ESI - IMCI margin per model/prompt/anchor. Higher = more ESI-leaning.

2. Logit-lens differential (declared noisy): per layer, project the residual
   stream through the final norm + lm_head and report (ESI-set first-token logit)
   - (IMCI-set first-token logit) at the post_action position. Comparative CURVE
   across layers; do not over-read single layers.

Run in `.venv-sft`. Greedy/teacher-forced; no transcoders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import cases as P

BASE = "Qwen/Qwen3-1.7B"
REPO = Path(__file__).resolve().parent.parent
DEF_R3 = REPO / "data/sft/checkpoints/qwen3-1.7b/adapter_r3"
DEF_R5 = REPO / "data/sft/checkpoints/qwen3-1.7b/adapter_r5"
ACTION_PREFIX = "Action: REFER NOW\n\n"


def load_model(kind, r3, r5):
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    if kind == "base":
        return base.eval()
    from peft import PeftModel
    return PeftModel.from_pretrained(base, r3 if kind == "r3" else r5
                                     ).merge_and_unload().eval()


def mean_token_logprob(model, tok, context_ids, phrase):
    pid = tok(phrase, add_special_tokens=False).input_ids
    if not pid:
        return None
    ids = torch.tensor([context_ids + pid], device="cuda")
    with torch.no_grad():
        logits = model(ids).logits[0]
    total = 0.0
    for i, t in enumerate(pid):
        lp = F.log_softmax(logits[len(context_ids) + i - 1].float(), -1)[t]
        total += float(lp)
    return total / len(pid)


def set_logprob(model, tok, context_ids, phrases):
    vals = [mean_token_logprob(model, tok, context_ids, ph) for ph in phrases]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals)  # mean over the set


def logit_lens_diff(model, tok, context_ids):
    """Per-layer (ESI first-token logit) - (IMCI first-token logit) at last pos."""
    ids = torch.tensor([context_ids], device="cuda")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple [n_layers+1] of [1,seq,d]
    norm = model.model.norm
    head = model.lm_head
    esi_first = [tok(p, add_special_tokens=False).input_ids[0] for p in P.ESI_PHRASES]
    imci_first = [tok(p, add_special_tokens=False).input_ids[0] for p in P.IMCI_PHRASES]
    curve = []
    for L, h in enumerate(hs):
        with torch.no_grad():
            lg = head(norm(h[0, -1])).float()
        esi = sum(float(lg[t]) for t in esi_first) / len(esi_first)
        imci = sum(float(lg[t]) for t in imci_first) / len(imci_first)
        curve.append(round(esi - imci, 3))
    return curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "data/eval/interp"))
    ap.add_argument("--adapter-r3", default=str(DEF_R3))
    ap.add_argument("--adapter-r5", default=str(DEF_R5))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    token_prob: dict = {}
    logit_lens: dict = {}

    for kind in ("base", "r3", "r5"):
        model = load_model(kind, args.adapter_r3, args.adapter_r5)
        for pid in P.PROMPTS:
            rendered = tok.apply_chat_template(P.chat_messages(pid), tokenize=False,
                                               add_generation_prompt=True)
            start_ids = tok(rendered, add_special_tokens=False).input_ids
            post_ids = tok(rendered + ACTION_PREFIX, add_special_tokens=False).input_ids
            for anchor, ctx in (("start", start_ids), ("post_action", post_ids)):
                esi = set_logprob(model, tok, ctx, P.ESI_PHRASES)
                imci = set_logprob(model, tok, ctx, P.IMCI_PHRASES)
                token_prob.setdefault(pid, {}).setdefault(anchor, {})[kind] = {
                    "esi_set_logprob": round(esi, 3),
                    "imci_set_logprob": round(imci, 3),
                    "esi_minus_imci": round(esi - imci, 3),
                }
            logit_lens.setdefault(pid, {})[kind] = logit_lens_diff(model, tok, post_ids)
        del model
        torch.cuda.empty_cache()

    (out / "token_prob.json").write_text(json.dumps(token_prob, indent=2), encoding="utf-8")
    (out / "logit_lens.json").write_text(json.dumps(logit_lens, indent=2), encoding="utf-8")

    print("=== ESI - IMCI margin (post_action anchor; higher = more ESI-leaning) ===")
    for pid in P.PROMPTS:
        row = token_prob[pid]["post_action"]
        print(f"{pid:16s}  " + "  ".join(
            f"{k}:{row[k]['esi_minus_imci']:+.2f}" for k in ("base", "r3", "r5")))
    print(f"\nwrote {out/'token_prob.json'} + {out/'logit_lens.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
