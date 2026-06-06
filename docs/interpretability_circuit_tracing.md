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

Full numbers in `data/eval/interp/REPORT.md` (token_prob.json, logit_lens.json,
outputs.json). Observations:

**1. Control gate — the leak is fine-tune-introduced, not inherited.** On all
trace prompts the BASE model's output frames in IMCI/ETAT (driven by the system
prompt; on the adult case it even over-applies IMCI). The token-set margin (ESI −
IMCI mean logprob; more negative = more IMCI-leaning) is most negative for base
everywhere. So the base does NOT carry an ESI prior on these prompts → per the
control gate, circuit-tracer on the base is context-only and the weight shifts to
the transcoder-free fine-tune comparison.

**2. Single-turn deployed behaviour is largely clean.** On these minimal,
realistic single-turn prompts r3 and r5 lead `Action: REFER NOW`; r5 explicitly
emits "No ESI or SATS code is assigned" on `esi_leak_resp` and `mietic_vitals`.
The leak does NOT reproduce single-turn. The Bloom 9.28 is therefore driven by the
ADVERSARIAL MULTI-TURN scenarios (vitals injected over turns), not normal
single-turn use.

**3. ESI propensity (token-set logprob) — fine-tuning added it; round-5 suppresses
it conditionally.** Margin base -> r3 moves toward ESI (e.g. `mietic_vitals` start
−3.22 -> −0.98). The round-5 effect is ANCHOR-DEPENDENT:
- after `Action: REFER NOW`: r5 is MORE IMCI than r3 (`esi_leak_resp` −3.42 vs
  −2.18; `mietic_vitals` −2.09 vs −0.10) — the round-5 "No ESI/SATS" line is a
  learned, conditional suppressor;
- at answer start: r5 is NOT lower than r3 (even higher ESI lean, e.g.
  `esi_leak_resp` −1.36 vs −2.52) — the general prior persists before the action.

**4. Logit-lens (noisy).** The ESI − IMCI direction is positive in late layers and
is amplified by fine-tuning vs base (peak e.g. `esi_leak_resp` base +1.44@L0 ->
r5 +4.26@L27); the peak deepens to layers 27-28 in the fine-tunes.

**5. circuit-tracer base graphs.** Context-only (base carries no ESI prior).
Saved (gitignored) under `data/eval/interp/graphs/`.

## Interpretation

The analysis suggests that the ESI-label prior remains behaviorally strong in
fine-tuned models; circuit tracing on the base model is used only to inspect
possible origins of this prior. The prior is introduced by the MIETIC fine-tuning
(absent in base); round-5 suppresses it CONDITIONALLY (after the action line),
which matches the clean single-turn behaviour, but does not remove the general
propensity — consistent with the residual leak appearing under Bloom's adversarial
multi-turn framing rather than in single-turn use. Observations are separated from
these weak hypotheses; no safety claim is made (the model still leads
`Action: REFER NOW` on danger signs).

## Limits

Transcoders base-only (attribution graphs not computed on the fine-tunes, by
design); fine-tune evidence is correlational. nnsight experimental/slower; Qwen3
unsupported in TransformerLens. PLT (per-layer) not cross-layer. Logit-lens noisy.
Activation patching (if used) is strong only within-model; cross-model is
exploratory (representations shift with fine-tuning). bf16 weights traced vs
Q4_K_M deployed. Observational only; small 1.7B model; possible base-vs-instruct
transcoder mismatch (verified at run time).
