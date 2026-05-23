# Preprocessing Model Selection Study

## Scope

Steps 2 through 4 use an LLM as an offline ETL component. The LLM extracts structured fields, rewrites records into questions, and paraphrases identity prompts. These tasks are not fine-tuning jobs.

| Step | LLM operation | Approximate calls | Difficulty |
| --- | --- | ---: | --- |
| 2: data cleaning | structured extraction from MIETIC narratives | 9.6k | low |
| 3: Q&A rewrite | style-specific question generation | 30k | medium |
| 4: identity augmentation | short paraphrases | 144 | low |

All generation uses guided JSON decoding where possible. Syntax is constrained by schema; field quality still depends on the model.

## Selection Criteria

- JSON adherence and semantic field quality.
- Offline batch throughput under vLLM.
- Compatibility with the available CUDA/vLLM stack.
- BF16 memory footprint on a single A100 80 GB or L40S 46 GB.
- License compatibility.
- Stable tokenizer/chat behavior for downstream Q&A data.

## Candidate Summary

| Model | Parameters | BF16 size | Context | License | vLLM 0.9.2 | Notes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Qwen3-1.7B | 1.7B | 3.4 GB | 32k | Apache-2.0 | yes | fastest option; semantic extraction risk |
| Qwen3-4B | 4B | 8 GB | 32k | Apache-2.0 | yes | strong quality/speed tradeoff |
| Qwen3-8B | 8.2B | 16 GB | 32k | Apache-2.0 | yes | robust ETL candidate with high throughput |
| Qwen3-14B | 14B | 29.6 GB | 32k | Apache-2.0 | yes | higher quality, lower throughput |
| Qwen3-32B | 32B | 62 GB | 32k | Apache-2.0 | yes | selected implementation model |
| Qwen3-30B-A3B | 30B MoE | about 60 GB | 32k | Apache-2.0 | partial | router/runtime complexity |
| Qwen3.6 family | 27B+ | 54 GB+ | 262k+ | Apache-2.0 | no | requires newer vLLM than the server supported |
| Llama-3.1-8B-Instruct | 8B | 16 GB | 128k | Llama license | yes | less reliable structured JSON behavior than Qwen in this pipeline |
| Mistral-Small-24B | 24B | 48 GB | 32k | Apache-2.0 | yes | viable but heavier than needed |
| Phi-4-14B | 14B | 28 GB | 16k | MIT | yes | tighter context and stricter safety behavior |

## Implemented Choice

The preprocessing stages used **Qwen3-32B BF16** through vLLM 0.9.2.

Operational reasons:

- The server CUDA driver constrained vLLM to the 0.9.x line.
- Qwen3.6 models were excluded by runtime compatibility.
- Qwen3-32B fit on a single A100 80 GB without tensor parallelism.
- Qwen-family chat and JSON behavior aligned with the rest of the pipeline.

Observed sizing:

- about 61 GiB weights
- about 4.7 GiB KV cache in the configured run
- continuous-batching concurrency about 2.3x at 8k context and higher on the shorter actual payloads
- MIETIC extraction runtime about 1.5 to 2 hours

## Lower-Cost Alternative

Qwen3-8B is the preferred lower-cost alternative for future reruns of the same preprocessing workload. The task profile is mostly slot filling and style-controlled question generation, so the marginal quality gain from 32B is limited relative to throughput cost.

Expected advantages of Qwen3-8B for this workload:

- lower BF16 memory footprint
- substantially higher vLLM concurrency
- faster end-to-end batch processing
- same Qwen-family tokenizer and chat conventions
- Apache-2.0 license

## Server Stack

Validated stack:

```text
Python 3.10.12
vLLM 0.9.2
torch 2.7.0+cu126
transformers 4.52.4
VLLM_USE_V1=0
VLLM_ATTENTION_BACKEND=XFORMERS
```

A local vLLM compatibility patch was applied for an `aimv2` config registration conflict in the server environment.

## Decision Record

| Field | Value |
| --- | --- |
| Implemented model | Qwen3-32B BF16 |
| Preferred future ETL model | Qwen3-8B |
| Runtime | vLLM offline batch |
| Main constraint | CUDA/vLLM version compatibility |
| Main cost | lower throughput from the 32B dense model |
