# Phase 2 Bloom Evaluation Study

## Scope

Phase 2 builds an advanced, reproducible **behavioral** evaluation before
scaling models (Phase 3). The static evaluators in `6_evaluation/` score fixed
prompts with regex; they cannot probe the high-stakes triage failure modes
systematically. This study integrates **Bloom** (Meridian Labs, on the
`inspect_ai` substrate — the same family as Petri) to auto-generate behavioral
test suites and score them with an LLM judge. Petri is the next step and reuses
this footprint.

## Limits of the current (static) evaluation

- Everything is static: fixed validation prompts, 10 identity probes, 17
  emergent probes, fixed QA benchmarks. Coverage of the behavior space is blind.
- Scoring is regex/keyword: brittle, no semantic judge (emergent probes already
  need manual review).
- No domain behavioral safety: refusal rate is a crude proxy; nothing tests
  false reassurance, failure to escalate, sycophancy, or fabricated dosing.
- No versioned seed/config for behaviors; mostly single-turn while triage is
  multi-turn.

## How Bloom works

Four-stage agentic pipeline: **Understanding → Ideation → Rollout → Judgment**.
Input is one behavior defined by a native `BEHAVIOR.md` (YAML frontmatter:
`num_scenarios`, `modality` = `conversation`|`agent`, optional `variations` such
as `noise`/`emotional_pressure`, plus a natural-language description). Bloom
generates many scenarios, runs multi-turn auditor↔target rollouts, and an LLM
**judge** scores behavior frequency/severity plus `eval_awareness` and
`scenario_realism`. Output is native `inspect_ai` `.eval` logs. Reproducibility =
cite the seed. Models are addressed with `inspect_ai` model strings and the
`--model-role auditor=/target=/judge=` flags.

Critical findings: the original `safety-research/bloom` is frozen; the live home
is Meridian `petri-bloom` (PyPI), same `inspect_ai` substrate as Petri. Stock
behaviors are alignment ones, so our triage behaviors are authored as custom
seeds.

## Integration strategies considered

- **A. Minimal** — run the `bloom` CLI by hand against local endpoints. Fastest,
  zero risk, but siloed and not comparable. Good only as the spike.
- **B. Thin wrapper subpackage on Bloom's native standard (chosen).** A
  self-contained `6_evaluation/behavioral/` that configures local endpoints,
  keeps native `.eval` output, and adds a cross-model reader. Additive, modular,
  reproducible, Petri-ready.
- **C. Full `inspect_ai` migration** — re-platform all evaluators. Big rewrite,
  high risk; rejected for this phase.

## Chosen architecture (Strategy B)

```
6_evaluation/
├── common.py                shared by static + behavioral
├── requirements.txt         extended: + petri-bloom (pulls inspect_ai)
├── static_evaluation/       existing evaluators (moved, unchanged)
└── behavioral/
    ├── endpoints.py         local endpoint config -> inspect model-role strings (one place to swap local<->API)
    ├── serve_model.sh       serve a checkpoint as a vLLM OpenAI endpoint with Bloom's required flags
    ├── run_behavioral.py    thin CLI: configure endpoints + bloom scenarios + inspect eval (native output)
    ├── compare_behavioral.py  read native .eval logs -> cross-model table
    └── seeds/<behavior>/BEHAVIOR.md   native, versioned behavior definitions

data/eval/behavioral/<model>/<behavior>/   native inspect .eval logs (no custom schema)
```

Design rules: existing evaluators untouched; one endpoint config; native
artifacts only; seeds versioned in-repo; Petri drops in beside this, reusing the
endpoints, the venv, and the native logs.

## Behaviors (first set, all 4 authored)

- **false_reassurance** — minimises a red-flag presentation.
- **failure_to_escalate** — under-triages a danger-sign presentation.
- **sycophantic_agreement** — defers to a caregiver's wrong framing.
- **fabricated_dosing** — gives an unsourced drug dose without clinician hand-off.

## Local auditor/judge decision

Per the user, the auditor + judge run **local** (no external API): a strong
local model (e.g. Qwen3-8B / Qwen3.6-27B) served via vLLM. This bounds judge
quality and is the main accuracy caveat; `endpoints.py` keeps each role's base
URL/key swappable so an API model can replace the local one later without code
changes.

## Spike findings (verified end-to-end, local only)

A throwaway run confirmed the full pipeline against local vLLM endpoints: `bloom
scenarios` → `inspect eval petri_bloom/bloom_audit` produced scored native
`.eval` logs (behavior + `eval_awareness` + `scenario_realism`). Operational
requirements baked into `serve_model.sh`:

- vLLM must run with `--enable-auto-tool-choice --tool-call-parser hermes` (Qwen
  tool calling) and `--max-model-len 32768` (Bloom requests up to 8192
  completion tokens).
- The target served from our merged BF16 checkpoint hit a tokenizer
  incompatibility under the pinned `transformers` in the vLLM env; the target is
  therefore served from its **deployed Q4_K_M GGUF via `llama-server`** (an
  OpenAI-compatible endpoint), which also means we evaluate the artifact we ship.

## Reproducibility & comparison

Each behavior's seed (`BEHAVIOR.md`) and generated scenarios are versioned in the
repo. Results are native `.eval` logs under `data/eval/behavioral/`, viewable
with `inspect view`. `compare_behavioral.py` builds a per-behavior cross-model
table from those logs without touching the static composite.

## Petri-readiness

Petri (same Meridian / `inspect_ai` family) integrates next: it reuses
`endpoints.py` and the `.venv-bloom`, and consumes these Bloom outputs as input.
No rewrite required.
