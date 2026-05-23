# Step 6: Evaluation

## Scope

Step 6 evaluates the merged candidate checkpoints from Step 5 and selects the primary model for GGUF deployment.

Input:

- `data/sft/checkpoints/<slug>/merged/`
- `data/augment/validation.jsonl`

Output:

- `data/eval/internal/<slug>.json`
- `data/eval/external/<slug>.json`
- `data/eval/qualitative/<slug>_samples.jsonl`
- `data/eval/summary.json`

## Architecture

```text
6_evaluation/
|-- common.py             # model loading, generation, shared paths
|-- prompts.py            # system prompt, identity probes, parsers
|-- eval_internal.py      # validation PPL, ESI/SATS accuracy, refusal, latency
|-- eval_external.py      # MedQA, PubMedQA, MMLU clinical
|-- eval_qualitative.py   # 30 stratified samples per model
|-- summarize.py          # composite ranking and gate checks
`-- requirements.txt
```

## Internal Evaluation

`eval_internal.py` scores each model on the Step 4 validation set.

| Metric | Definition |
| --- | --- |
| Validation NLL / perplexity | token-level loss on validation records |
| ESI accuracy | parsed `ESI N` match against gold |
| SATS accuracy | parsed `SATS <color>` match against gold |
| Pediatric recall | keyword-overlap recall on WHO IMCI / WHO ETAT records |
| Refusal rate | refusal classifier over 10 identity probes |
| Latency | generated tokens per second on BF16 GPU |
| Style breakdown | ESI/SATS accuracy by `metadata.style` |

The refusal metric counts only plain refusals without actionable escalation content. A disclaimer plus a useful next step is not counted as refusal.

## External Evaluation

`eval_external.py` runs lightweight medical QA benchmarks from Hugging Face:

| Benchmark | Split / subset | Default sample size | Metric |
| --- | --- | ---: | --- |
| MedQA-USMLE | English test | 100 | 4-choice accuracy |
| PubMedQA | `pqa_labeled` | 100 | yes/no/maybe accuracy |
| MMLU clinical knowledge | test | 50 | 4-choice accuracy |

External QA scores are reported but not included in the deployment composite because they are general medical exam tasks rather than LMIC triage tasks.

## Qualitative Evaluation

`eval_qualitative.py` samples up to five validation records per rewrite style and writes model responses next to the gold answer. These files support manual review for clinical plausibility, tone, pediatric content, and unsupported claims.

## Composite Ranking

`summarize.py` ranks candidates with the following weights:

```text
composite = 0.25 * esi_accuracy
          + 0.20 * pediatric_recall
          + 0.20 * non_refusal_rate
          + 0.20 * latency_score
          + 0.15 * sats_accuracy
```

`latency_score = min(tokens_per_s / 50, 1.0)`.

Hard gates:

| Gate | Threshold |
| --- | ---: |
| Minimum BF16 GPU latency | 10 tok/s |
| Maximum refusal rate | 30% |
| Minimum pediatric recall | 0.40 |

## Execution

Run each candidate on the desired GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python 6_evaluation/eval_internal.py --model qwen3-1.7b
CUDA_VISIBLE_DEVICES=1 python 6_evaluation/eval_internal.py --model smollm3-3b
CUDA_VISIBLE_DEVICES=5 python 6_evaluation/eval_internal.py --model gemma-3n-e2b
```

External benchmarks:

```bash
CUDA_VISIBLE_DEVICES=0 python 6_evaluation/eval_external.py --model qwen3-1.7b
CUDA_VISIBLE_DEVICES=1 python 6_evaluation/eval_external.py --model smollm3-3b
CUDA_VISIBLE_DEVICES=5 python 6_evaluation/eval_external.py --model gemma-3n-e2b
```

Qualitative samples:

```bash
CUDA_VISIBLE_DEVICES=0 python 6_evaluation/eval_qualitative.py --model qwen3-1.7b
CUDA_VISIBLE_DEVICES=1 python 6_evaluation/eval_qualitative.py --model smollm3-3b
CUDA_VISIBLE_DEVICES=5 python 6_evaluation/eval_qualitative.py --model gemma-3n-e2b
```

Aggregate the ranking:

```bash
python 6_evaluation/summarize.py
```

## Result Summary

| Model | ESI | SATS | Pediatric recall | Refusal rate | BF16 tok/s | Gate | Composite |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Qwen3-1.7B | 0.928 | 0.928 | 0.7867 | 0.000 | 59.89 | pass | 0.9285 |
| SmolLM3-3B | 0.948 | 0.948 | 0.3400 | 0.000 | 64.34 | fail pediatric | 0.8472 |
| Gemma-3n-E2B-it | 0.000 | 0.000 | 0.5500 | 0.000 | 22.00 | pass | 0.3980 |

Decision: `qwen3-1.7b` is the Phase 1 primary model for deployment.

## Out of Scope

- GGUF conversion and quantized evaluation.
- Android device validation.
- DPO or preference tuning.
