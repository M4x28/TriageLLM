# Behavioral evaluation (Bloom)

Phase 2 advanced evaluation.

Everything runs **local**: no external API. The target model and the
auditor/judge model are each served behind a vLLM OpenAI-compatible endpoint;
Bloom reaches them with `inspect_ai` service-prefixed model strings.

## Layout

```
behavioral/
├── endpoints.py        local endpoint config + inspect model-role strings (one place to swap local<->API)
├── serve_model.sh      serve a checkpoint as a vLLM OpenAI endpoint with the flags Bloom needs
├── run_behavioral.py   thin CLI: configure endpoints + bloom scenarios + inspect eval (native output)
├── compare_behavioral.py   read native .eval logs -> cross-model table
├── seeds/<behavior>/BEHAVIOR.md   native Bloom behavior definitions (versioned, reproducible)
└── (deps live in ../requirements.txt; installed in a dedicated .venv-bloom)
```

Results are Bloom/`inspect_ai` **native artifacts** (`.eval` logs) — no custom
schema. They are collected under `data/eval/behavioral/<model>/<behavior>/`.

## Setup (once, on the server)

```bash
# 1. Bloom orchestrator venv (separate from the vLLM serving env)
python3 -m venv .venv-bloom && source .venv-bloom/bin/activate
pip install petri-bloom

# 2. Serve the auditor/judge (strong local model) and the target (model under test).
#    serve_model.sh applies the required flags (tool-choice + 32k context).
bash 6_evaluation/behavioral/serve_model.sh Qwen/Qwen3-8B qwen3-8b 8002 2          # auditor/judge
bash 6_evaluation/behavioral/serve_model.sh data/sft/checkpoints/qwen3-1.7b/merged qwen3-1.7b 8001 0   # target
```

## Run a behavior

```bash
source .venv-bloom/bin/activate
python 6_evaluation/behavioral/run_behavioral.py --model qwen3-1.7b --behavior false_reassurance
```

This configures the endpoints, runs `bloom scenarios` (Understanding + Ideation)
then `inspect eval petri_bloom/bloom_audit` (Rollout + Judgment), and copies the
native `.eval` log into `data/eval/behavioral/<model>/<behavior>/`.

View any run with the native viewer: `inspect view --log-dir <dir>`.

Compare models on a behavior:

```bash
python 6_evaluation/behavioral/compare_behavioral.py --behavior false_reassurance
```

## Add a behavior

Create `seeds/<name>/BEHAVIOR.md` (YAML frontmatter + a natural-language
description; see `seeds/false_reassurance/BEHAVIOR.md`). Frontmatter fields:
`num_scenarios`, `modality` (`conversation`|`agent`), optional `variations`
(e.g. `noise`, `emotional_pressure`), `instructions`, `target_sysprompt_prefix`.
Then run it as above.

## Notes

- The vLLM server MUST run with `--enable-auto-tool-choice --tool-call-parser
  hermes` (Qwen tool calling) and `--max-model-len 32768` (Bloom requests up to
  8192 completion tokens). `serve_model.sh` sets these.