# Step 5: SFT Training

## Scope

Step 5 trains LoRA adapters for the three Phase 1 candidate models using the Step 4 SFT dataset.

Input:

- `data/augment/train.jsonl`
- `data/augment/validation.jsonl`

Output:

- `data/sft/checkpoints/<slug>/adapter/`
- `data/sft/checkpoints/<slug>/merged/`

## Architecture

```text
5_sft_training/
|-- common.py         # model registry, dataset loader, shared paths
|-- train.py          # TRL SFTTrainer with PEFT LoRA
|-- merge_lora.py     # merge adapter into base model
`-- requirements.txt
```

The stage imports JSONL and logging helpers from the cleaning stage through `importlib.util`.

## Candidate Models

| Slug | Base model | Parameters | License | Role |
| --- | --- | ---: | --- | --- |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 1.7B | Apache-2.0 | primary mobile candidate |
| `smollm3-3b` | `HuggingFaceTB/SmolLM3-3B` | 3B | Apache-2.0 | quality fallback |
| `gemma-3n-e2b` | `google/gemma-3n-E2B-it` | 2B effective / 6B raw | Gemma | mobile-optimized architecture candidate |

The model registry is defined in `5_sft_training/common.py`.

## LoRA Configuration

Default `train.py` arguments:

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    qwen3-1.7b: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    smollm3-3b: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    gemma-3n-e2b: [q_proj, k_proj, v_proj, o_proj]
training:
  learning_rate: 2.0e-4
  epochs: 3
  max_seq_length: 2048
  gradient_accumulation_steps: 8
  warmup_ratio: 0.03
  scheduler: cosine
  weight_decay: 0.01
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: true
  eval_steps: 200
  save_steps: 200
  seed: 42
```

Gemma-3n uses attention-only LoRA targets because its MatFormer and per-layer embedding structure is non-standard relative to the Qwen and SmolLM dense transformer blocks.

## Execution

Single-model training:

```bash
CUDA_VISIBLE_DEVICES=0 python 5_sft_training/train.py --model qwen3-1.7b
CUDA_VISIBLE_DEVICES=1 python 5_sft_training/train.py --model smollm3-3b
CUDA_VISIBLE_DEVICES=5 python 5_sft_training/train.py --model gemma-3n-e2b
```

Merge each adapter after training:

```bash
python 5_sft_training/merge_lora.py --model qwen3-1.7b
python 5_sft_training/merge_lora.py --model smollm3-3b
python 5_sft_training/merge_lora.py --model gemma-3n-e2b
```

## Output Layout

```text
data/sft/checkpoints/<slug>/
|-- adapter/
|   |-- adapter_config.json
|   |-- adapter_model.safetensors
|   |-- chat_template.jinja
|   |-- tokenizer.*
|   `-- training_args.bin
`-- merged/
    |-- config.json
    |-- generation_config.json
    |-- model.safetensors
    |-- tokenizer.*
    `-- chat_template.jinja
```

## Training Results

| Model | Total steps | Runtime | Validation perplexity |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B | 3,150 | about 2h45m | 1.51 |
| SmolLM3-3B | 6,297 | about 4h00m | 1.26 |
| Gemma-3n-E2B-it | 6,297 | about 5h57m | 12.32 |

Qwen3-1.7B and SmolLM3-3B converged cleanly on the SFT format. Gemma-3n-E2B-it retained high validation perplexity with attention-only LoRA and later failed the explicit triage-label metrics.

## Completion Checks

- `adapter/` exists for each trained model.
- `merged/` exists for each candidate used by Step 6.
- The merged tokenizer contains the expected chat template.
- Training and validation losses are recorded in the trainer output.
