"""Textual baseline + decision positions for base / round-3 / round-5.

Greedy-generates the full answer from each model for each trace prompt (so every
internal signal later is tied to REAL behaviour), records whether the framework
leaked (ESI markers on an under-5 case), and saves candidate analysis positions
(first assistant token + the positions preceding each detected ESI/IMCI phrase)
with top-k next-token logits.

All three models share the BASE tokenizer + chat template (adapters add no
tokens); the script asserts this. Run in `.venv-sft` (transformers + peft).

Usage (server):
  CUDA_VISIBLE_DEVICES=1 python interpretability/generate_outputs.py \
    --out-dir data/eval/interp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import cases as P

REPO = Path(__file__).resolve().parent.parent
DEF_R3 = REPO / "data/sft/checkpoints/qwen3-1.7b/adapter_r3"
DEF_R5 = REPO / "data/sft/checkpoints/qwen3-1.7b/adapter_r5"


def load_model(base_id: str, kind: str, path: str | None):
    """`path` is None (pure base), a LoRA adapter dir, or a full merged checkpoint dir."""
    if kind == "base" or path is None:
        return AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    if (Path(path) / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                     device_map="cuda")
        from peft import PeftModel
        return PeftModel.from_pretrained(base, path).merge_and_unload().eval()
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                device_map="cuda").eval()


def framework_of(text: str) -> str:
    low = text.lower()
    esi = any(m in low for m in P.ESI_MARKERS)
    imci = any(m in low for m in P.IMCI_MARKERS)
    if esi and not imci:
        return "ESI"
    if imci and not esi:
        return "IMCI"
    if esi and imci:
        return "MIXED"
    return "NEITHER"


def find_phrase_positions(tok, gen_ids: list[int], phrases: list[str]) -> list[int]:
    """Sequence indices (within gen_ids) where any phrase's first token starts."""
    hits = []
    low_ids = gen_ids
    for ph in phrases:
        pid = tok(ph, add_special_tokens=False).input_ids
        if not pid:
            continue
        for i in range(len(low_ids) - len(pid) + 1):
            if low_ids[i:i + len(pid)] == pid:
                hits.append(i)
    return sorted(set(hits))


def topk_at(logits_row, tok, k=8):
    vals, idx = torch.topk(logits_row.float(), k)
    return [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(vals, idx)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "data/eval/interp"))
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--kinds", default="base,r3,r5",
                    help="comma-separated kind names; 'base' needs no path")
    ap.add_argument("--kind-path", action="append", default=[],
                    metavar="NAME=PATH", help="override a kind's checkpoint/adapter dir")
    ap.add_argument("--adapter-r3", default=str(DEF_R3))
    ap.add_argument("--adapter-r5", default=str(DEF_R5))
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    kinds = args.kinds.split(",")
    paths = {"r3": args.adapter_r3, "r5": args.adapter_r5}
    for kv in args.kind_path:
        name, _, path = kv.partition("=")
        paths[name] = path

    tok = AutoTokenizer.from_pretrained(args.base)
    base_template = tok.chat_template

    outputs: dict = {}
    positions: dict = {}

    for kind in kinds:
        model = load_model(args.base, kind, paths.get(kind))
        # integrity: same tokenizer/template/dtype (we use the base tokenizer for
        # all; assert dtype bf16 and template unchanged).
        assert tok.chat_template == base_template, "chat template drift!"
        assert next(model.parameters()).dtype == torch.bfloat16, "dtype drift!"

        for pid, spec in P.PROMPTS.items():
            msgs = P.chat_messages(pid)
            rendered = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
            enc = tok(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
            plen = enc.input_ids.shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None,
                                     pad_token_id=tok.eos_token_id)
            full_ids = gen[0].tolist()
            gen_ids = full_ids[plen:]
            text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            fw = framework_of(text)
            outputs.setdefault(pid, {})[kind] = {
                "framework": fw,
                "expected": P.EXPECTED_MATRIX[pid],
                "text": text,
            }

            # candidate positions: first assistant token + before each ESI/IMCI phrase
            esi_pos = find_phrase_positions(tok, gen_ids, P.ESI_PHRASES)
            imci_pos = find_phrase_positions(tok, gen_ids, P.IMCI_PHRASES)
            cand = sorted(set([0] + esi_pos + imci_pos))
            with torch.no_grad():
                logits = model(gen).logits[0]  # [seq, vocab]
            recs = []
            for gp in cand[:12]:
                seq_index = plen + gp  # position predicting gen token gp
                pred_row = logits[seq_index - 1]
                recs.append({
                    "gen_offset": gp,
                    "kind": ("first_assistant" if gp == 0 else
                             ("pre_esi" if gp in esi_pos else "pre_imci")),
                    "topk": topk_at(pred_row, tok),
                })
            positions.setdefault(pid, {"rendered_prompt": rendered,
                                       "prompt_len": plen})[kind] = {
                "esi_phrase_offsets": esi_pos, "imci_phrase_offsets": imci_pos,
                "positions": recs,
            }
        del model
        torch.cuda.empty_cache()

    (out / "outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    (out / "decision_positions.json").write_text(json.dumps(positions, indent=2),
                                                 encoding="utf-8")
    # console leak summary
    print("=== framework per prompt/model (expected vs got) ===")
    for pid in P.PROMPTS:
        print(f"\n{pid}  [expect: {P.EXPECTED_MATRIX[pid]}]")
        for kind in kinds:
            print(f"  {kind:4s} -> {outputs[pid][kind]['framework']}")
    print(f"\nwrote {out/'outputs.json'} + {out/'decision_positions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
