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

**Judge upgrade (round-2 re-eval): Qwen3-8B -> Qwen3-32B.** The first Phase D
pass used Qwen3-8B as auditor/judge. Its scoring proved unreliable on the
auto-named per-behavior dimensions: polarity was inconsistent (the same "held the
referral firmly" behavior was scored both 1 and 10 on `caregiver_resists_referral`),
so the raw means could not be trusted and the failure reasons had to be read from
the transcripts. For the round-2 re-evaluation we therefore switched the
auditor/judge to **Qwen3-32B**, served via vLLM tensor-parallel across two L40S
(`serve_judge32b.sh`, `NCCL_P2P_DISABLE=1` to avoid the non-NVLink PCIe-P2P hang,
`--enforce-eager`). Qwen3-32B is the same Qwen3 architecture as the 8B (so the
`hermes` tool-call parser and transformers 4.52.4 / vLLM 0.9.2 stack work
unchanged); Qwen3.6-27B was tried first but is architecture `qwen3_5`, which the
pinned stack does not load. The 32B judge gives consistent polarity and is the
reference judge for the round-2 numbers.

## Reproducibility & comparison

Each behavior's seed (`BEHAVIOR.md`) and generated scenarios are versioned in the
repo. Results are native `.eval` logs under `data/eval/behavioral/`, viewable
with `inspect view`. `compare_behavioral.py` builds a per-behavior cross-model
table from those logs without touching the static composite.

## Risk matrix

| failure mode                   | clinical consequence                                                               | severity          | status (% >= 8) |        priority |
| ------------------------------ | ---------------------------------------------------------------------------------- | ----------------- | --------------: | --------------: |
| caregiver_resists_referral     | caves to caregiver pressure, withdraws an indicated referral; emergency stays home | Critical (policy) |             60% |               1 |
| failure_to_escalate            | under-triages danger signs; delayed care, death                                    | Critical          |             53% |               2 |
| vomits_everything_but_no_fever | dismisses a danger sign for lack of fever; dehydration/sepsis missed               | Critical          |             47% |               3 |
| benign_fever_no_danger_sign    | over-triages a well child; wastes scarce capacity, erodes trust                    | Moderate          |             27% |               4 |
| non_clinical_admin_request     | forces a triage label on a non-clinical task                                       | Low (mis-scope)   |            60%† |               5 |
| ambiguous_convulsion_history   | (recognition) catches lay-described seizures                                       | Safe (passing)    |             n/a |         monitor |
| post_ictal_lethargy            | over-escalates a post-convulsion child                                             | Safe over-caution |             n/a | judge-to-review |

## Qwen3-8B publication gate: complete 3-seed re-eval (32B judge)

Qwen3-8B SFT checkpoint (candidate for `v1.1.0-qwen3-8b experimental`), all 3
critical-priority seeds, same `serve_judge32b.sh` Qwen3-32B judge and versioned
scenarios as the 1.7B baseline above (no `--regen-scenarios`). Two model states
evaluated: the original 8B SFT checkpoint, and the same checkpoint after a
targeted LoRA patch (see `docs/7_interpretability.md` for the mechanistic
diagnosis and patch design — root cause: round-5's "no ESI/SATS" suppressor was
only ever trained on the `REFER NOW` branch, never on the HOME CARE branch).

| seed | 1.7B (% >= 8) | 8B pre-patch (% >= 8) | 8B + HOME CARE patch (% >= 8) |
| --- | ---: | ---: | ---: |
| pediatric_low_resource_wrong_framework_esi_leak | n/a (round 3/4/5 mean 9.28-9.44/10) | 83.3% (n=18) | **16.7%** (n=18) |
| failure_to_escalate | 53% | 53.8% (n=13/15)\* | **40%** (n=15, complete) |
| caregiver_resists_referral | 60% | 33.3% (n=15) | 33.3% (n=15, unchanged) |

\* pre-patch `failure_to_escalate` lost 2/15 samples to judge context-length
overflow (32B judge transcript + scoring schema over the 32768-token
`--max-model-len`); the post-patch re-run completed all 15 with no overflow
(different rollout content after the patch happened to stay under the limit).

**Verdict per failure mode:**
- **framework-leak: large improvement, not resolved.** 83.3% -> 16.7% (~5x
  reduction in both mean score and % >= 8). The patch fixed the diagnosed root
  cause (uncovered HOME CARE branch) but a residual tail remains — likely
  adversarial multi-turn framings not covered by the 25 static single-turn
  patch examples used to fix it.
- **failure_to_escalate: modest improvement, still above the 1.7B baseline.**
  53.8% -> 40%, a partial reduction but not targeted by this patch (the 25
  patch examples were benign HOME CARE cases, not danger-sign recognition
  cases) — the improvement is likely incidental (LoRA patch nudging general
  triage behaviour) rather than a direct fix. Still the highest-severity
  failure mode in the matrix and not resolved.
- **caregiver_resists_referral: unchanged (33.3%), already improved from 1.7B's
  60% before this patch.** Expected — this patch did not touch caregiver-
  pressure scenarios at all.

**Publication recommendation: still not clean for an unqualified release.**
The framework-leak facet, previously the primary publish blocker at 83.3%, is
now a secondary concern at 16.7%. `failure_to_escalate` at 40% is now the
dominant residual risk and was not the target of this patch cycle. Given the
`v1.1.0-qwen3-8b experimental` label already signals non-production status,
publishing with this documented residual risk is a judgment call for the
user/NGO partner, not an automatic pass — the numbers here are the complete,
apples-to-apples evidence for that decision, not a go/no-go verdict on their
own.

**Decision (this round): DO NOT PUBLISH `v1.1.0-qwen3-8b experimental`.**
`failure_to_escalate` is under-triage of real danger signs (missed cardiac
alert, seizure risk, hypoglycemia, hemorrhage, depression — see seed scenarios
in `6_evaluation/behavioral/seeds/failure_to_escalate/`), not a framework-
purity or policy-compliance issue like the other two seeds. A 40% failure rate
on this axis is a patient-safety risk that "experimental" labeling and UI
disclaimers do not mitigate enough to justify release. Next step: a targeted
LoRA patch analogous to the HOME CARE fix, but built against the specific
`failure_to_escalate` danger-sign scenarios (missed cardiac alert, seizure
risk, neglected hypoglycemia, overlooked hemorrhage/depression signs — see the
seed's scenario list) rather than reusing the HOME CARE patch data, which did
not target this failure mode and only improved it incidentally. Re-run all 3
seeds again after that patch before revisiting publication.

### Operational note: Bloom run stability with the 32B judge

Three consecutive `failure_to_escalate` re-runs against the patched checkpoint
hung indefinitely (`inspect eval` reporting 5-10/15 samples completed, network
sockets stuck `CLOSE-WAIT`, no error) regardless of `--max-connections` (tried
10 and 4) — the failure was **time-based** (consistently ~25-30 min wall time
into the run, not tied to sample count or concurrency). Root cause not fully
diagnosed (suspected TCP idle/keep-alive timeout on the long-running
auditor<->target<->judge round trips); fixed operationally by adding
`--timeout 120 --time-limit 900` to the `inspect eval` invocation (per-request
API timeout + per-sample wall-clock cap), which let the 4th attempt complete
cleanly in 17:58. Recommended default for future Bloom re-evaluations on this
stack; not yet wired into `run_behavioral.py` as a CLI flag (currently a manual
override to the underlying `inspect eval` call).