# Cleanup manifest - Phase 2 (under-triage fix + behavioral eval)

Date: 2026-06-03. Removes only regenerable temp artifacts created during the
Phase 2 retrain + Bloom evaluation session. Nothing below affects results,
model artifacts, or other users' files.

## Deleted

### Server `/data2/lbirardi/TriageLLM`
| path | what | safe because |
| --- | --- | --- |
| `logs/gen_triage_seed.log`, `reaugment.log`, `resft_qwen3-1.7b.log`, `merge_qwen3-1.7b.log`, `gguf_convert.log`, `gateA_*.log`, `phaseD_*.log`, `target_llama.log`, `auditor_*.log` | session run logs | transient; key numbers captured in the docs |
| `data/deploy/gguf/qwen3-1.7b-f16.gguf` (~3.45 GB) | lossless F16 GGUF intermediate | only needed to produce Q4_K_M; regenerable from `merged/` via `7_deploy/convert_and_quantize.sh` |
| `data/augment/train.prev.jsonl`, `data/augment/validation.prev.jsonl` | pre-fix augment backups | regenerable from `data/rewrite/` with the old `augment.py` (git history) |
| `6_evaluation/behavioral/.work/*` | gitignored generated scenarios + injected BEHAVIOR copies | regenerated on each `run_behavioral.py` run |

### Local `c:\Projects\TriageLLM`
| path | what | safe because |
| --- | --- | --- |
| `6_evaluation/behavioral/.work/*` | gitignored injection test dir | regenerable |

## Kept (explicitly NOT deleted)
- `data/eval/behavioral/qwen3-1.7b/**` + `PHASE_D_SUMMARY.md` - results under analysis.
- `data/sft/checkpoints/qwen3-1.7b/adapter_v101/` - v1.0.1 LoRA adapter backup.
- `data/deploy/gguf/qwen3-1.7b-q4_k_m.gguf` (v2) and `...-q4_k_m.v101.gguf` (v1.0.1).
- `data/sft/checkpoints/qwen3-1.7b/merged/` - merged v2 checkpoint (source for GGUF).
- Any process/file owned by other users (e.g. `gmgori`'s `auto_infer.py`).
