# Circuit-tracing the residual ESI framework leak (research)

Research-only interpretability of the persistent `ESI Level` framework leak on
under-5 low-resource cases. Rounds 4-5 (data-side) did not move the Bloom seed
`pediatric_low_resource_wrong_framework_esi_leak` (9.28); the resource-prediction
facet was eliminated but the ESI-label facet persists. Safety is intact (the model
leads `Action: REFER NOW`). This phase asks *how* the leak arises internally — it
does NOT validate safety and does not justify a fix by itself.

## Method

- **Main: circuit-tracer attribution graphs on the BASE `Qwen/Qwen3-1.7B` only**
  (Qwen3-1.7B PLT transcoders `mwhanna/qwen3-1.7b-transcoders-lowl0` are
  base-trained; nnsight backend). Inspect which features feed the ESI vs IMCI
  decision on the trace prompts.
- **Fine-tune (base vs round-3 vs round-5): transcoder-free** — textual outputs,
  token-set sequence logprob (ESI-set vs IMCI-set, at the generation start and
  after `Action: REFER NOW`), and a per-layer logit-lens differential.
- **Control gate:** if the base shows no ESI prior on the pediatric prompts, the
  leak is born/reinforced by the MIETIC fine-tuning rather than inherited.

Expected-behavior matrix (the hypothesis): adult+ED+vitals -> ESI ok; under-5
low-resource (with or without an explicit ESI request) -> IMCI/ETAT, no ESI;
benign under-5 -> no ESI. The leak = ESI emitted for under-5 low-resource.

Code: `interpretability/`. Results: `data/eval/interp/` (`REPORT.md` + small JSON
committed; heavy `graphs/` gitignored). The clean-IMCI contrast is a NATURAL
clean generation at constant input, never a steered prompt.

## Results

_To be filled from `data/eval/interp/REPORT.md` after the run._

## Interpretation

The analysis suggests that the ESI-label prior remains behaviorally strong in
fine-tuned models; circuit tracing on the base model is used only to inspect
possible origins of this prior. Observations are separated from weak hypotheses;
no safety claim is made.

## Limits

Transcoders base-only (attribution graphs not computed on the fine-tunes, by
design); fine-tune evidence is correlational. nnsight experimental/slower; Qwen3
unsupported in TransformerLens. PLT (per-layer) not cross-layer. Logit-lens noisy.
Activation patching (if used) is strong only within-model; cross-model is
exploratory (representations shift with fine-tuning). bf16 weights traced vs
Q4_K_M deployed. Observational only; small 1.7B model; possible base-vs-instruct
transcoder mismatch (verified at run time).
