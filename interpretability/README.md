# Interpretability — ESI framework-leak (research)

Research-only investigation of the residual `ESI Level` framework leak on under-5
low-resource cases (rounds 4-5 did not move Bloom 9.28). **Not** safety
validation, **not** an auto-fix. No retrain / DPO / model change / deploy.

## Method (decided)

- **Main — circuit-tracer attribution graphs on the BASE `Qwen/Qwen3-1.7B` ONLY.**
  Its Qwen3-1.7B PLT transcoders (`mwhanna/qwen3-1.7b-transcoders-lowl0`) are
  base-trained, so they are valid only on the base. Backend = **nnsight**
  (TransformerLens does not support Qwen3).
- **Fine-tune (base vs round-3 vs round-5) — transcoder-free**, in order:
  (1) textual outputs, (2) token-set sequence logprob, (3) logit-lens
  differential; activation patching is deferred (`patching.py`, only if needed).
- **Control gate:** if the base shows no ESI prior on the pediatric prompts, the
  leak is born/reinforced by the MIETIC fine-tuning (a real finding, not a
  failure) and the weight shifts to the transcoder-free comparison.

## Two venvs

- `.venv-sft` (transformers + peft): `generate_outputs.py`, `finetune_diff.py`,
  `summarize_interp.py`. Loads base + applies `adapter_r3`/`adapter_r5` in memory
  (`merge_and_unload`); all three share the base tokenizer/chat template (asserted).
- `.venv-circuit` (circuit-tracer + nnsight): `trace_base.py`.

```bash
python3 -m venv .venv-circuit && source .venv-circuit/bin/activate
pip install circuit-tracer        # decoderesearch/circuit-tracer
```

## Run (server, GPU 1)

```bash
# 1. textual baseline + decision positions (control: does the leak reproduce? base prior?)
CUDA_VISIBLE_DEVICES=1 .venv-sft/bin/python interpretability/generate_outputs.py

# 2. transcoder-free fine-tune diff
CUDA_VISIBLE_DEVICES=1 .venv-sft/bin/python interpretability/finetune_diff.py

# 3. circuit-tracer attribution on base (pilot thresholds first)
CUDA_VISIBLE_DEVICES=1 .venv-circuit/bin/python interpretability/trace_base.py --pilot
CUDA_VISIBLE_DEVICES=1 .venv-circuit/bin/python interpretability/trace_base.py \
    --node-threshold 0.5 --edge-threshold 0.9
#    inspect one: ... trace_base.py --only esi_leak_resp --server   (viewer :8041)

# 4. report
.venv-sft/bin/python interpretability/summarize_interp.py
```

## Outputs

`data/eval/interp/`: `outputs.json`, `decision_positions.json`, `token_prob.json`,
`logit_lens.json`, `REPORT.md` (committed); `graphs/` heavy `.pt`/`.html`
(**gitignored** — keep local or Git LFS).

## Limits (declare in the report)

Transcoders base-only (no fine-tune graphs, by design). nnsight experimental /
slower. PLT not CLT (partial). Logit-lens noisy. Patching (if used) within-model
only as strong; cross-model exploratory (reps shift with fine-tuning). bf16 traced
vs Q4_K_M deployed. Observational; does NOT validate safety; small 1.7B. Verify
the transcoders are for base (not instruct) Qwen3-1.7B. Long SYSTEM_PROMPT makes
attribution graphs heavy — use `--offload cpu`.
