# Phase 2 Bloom Evaluation Study

## Scope

Phase 2 builds an advanced, reproducible **behavioral** evaluation before
scaling models (Phase 3). The static evaluators in `6_evaluation/` score fixed
prompts with regex; they cannot probe the high-stakes triage failure modes
systematically. This study integrates **Bloom** (Meridian Labs, Petri) to auto-generate behavioral
test suites and score them with an LLM judge. Petri is the next step and reuses this footprint.

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
cite the seed.

## Integration strategies considered

- **A. Minimal**: run the `bloom` CLI by hand against local endpoints. Fastest,
  zero risk, but siloed and not comparable. Good only as the spike.
- **B. Thin wrapper subpackage on Bloom's native standard (chosen).** A
  self-contained `6_evaluation/behavioral/` that configures local endpoints,
  keeps native `.eval` output, and adds a cross-model reader. Additive, modular,
  reproducible, Petri-ready.

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

## Behaviors (first set, all 4 authored)

- **false_reassurance**: minimises a red-flag presentation.
- **failure_to_escalate**: under-triages a danger-sign presentation.
- **sycophantic_agreement**: defers to a caregiver's wrong framing.
- **fabricated_dosing**: gives an unsourced drug dose without clinician hand-off.

## Local auditor/judge decision

Per the user, the auditor + judge run **local** (no external API): a strong
local model e.g. Qwen3-8B / Qwen3.6-27B. This bounds judge quality and is the main accuracy caveat; `endpoints.py` keeps each role's basenURL/key swappable so an API model can replace the local one later without code changes.

## Reproducibility & comparison

Each behavior's seed (`BEHAVIOR.md`) and generated scenarios are versioned in the
repo. Results are native `.eval` logs under `data/eval/behavioral/`, viewable
with `inspect view`. `compare_behavioral.py` builds a per-behavior cross-model
table from those logs without touching the static composite.