# Fine-Tuning and Hyperparameter Study

## Scope

This study documents the fine-tuning method and hyperparameter configuration used for TriageLLM Phase 1.

Task profile:

- 33,575 SFT records in Hugging Face `messages` format.
- Models in the 1.7B to 3B range.
- BF16 training on NVIDIA L40S 46 GB GPUs.
- Deployment through merged checkpoint conversion to GGUF Q4_K_M.

## Fine-Tuning Options

| Method | Trainable parameters | Strengths | Limitations |
| --- | ---: | --- | --- |
| Full SFT | 100% | highest capacity | overfit risk and higher memory cost for a 33k-record dataset |
| LoRA | about 0.1% to 1% | mature, efficient, mergeable with no inference overhead | lower capacity than full SFT |
| QLoRA | about 0.1% to 1% | lower VRAM use | slower than BF16 LoRA when VRAM is available |
| DoRA | LoRA plus magnitude terms | better low-rank adaptation in some tasks | slower and more complex than LoRA |
| IA3 | very small | parameter efficient | likely underfits clinical behavior |
| Prompt/prefix tuning | very small | lightweight | weak for domain and safety behavior changes |
| Adapter modules | small | modular | inference latency overhead |
| DPO | preference post-training | useful when SFT behavior needs alignment | requires preference pairs |
| RLHF/PPO | policy optimization | powerful alignment method | excessive complexity for Phase 1 |

## Selected Method

Phase 1 uses **LoRA SFT in BF16**.

Rationale:

- Good fit for 1.7B to 3B models on L40S GPUs.
- Lower overfit risk than full SFT on the available dataset.
- Mature support in PEFT and TRL.
- Adapter can be merged into the base model before GGUF conversion.
- No inference latency overhead after merge.

QLoRA is not used because memory is sufficient for BF16 LoRA and the final deployment format is GGUF, not NF4.

## Training Configuration

```yaml
peft: LoRA
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
training:
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler: cosine
  num_train_epochs: 3
  max_seq_length: 2048
  weight_decay: 0.01
  max_grad_norm: 1.0
  gradient_accumulation_steps: 8
  bf16: true
  gradient_checkpointing: true
  eval_strategy: steps
  eval_steps: 200
  save_steps: 200
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
```

Per-device batch size is model-specific:

| Model | Per-device train batch |
| --- | ---: |
| Qwen3-1.7B | 4 |
| SmolLM3-3B | 2 |
| Gemma-3n-E2B-it | 2 |

## LoRA Target Modules

Qwen3-1.7B and SmolLM3-3B:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Gemma-3n-E2B-it:

```text
q_proj, k_proj, v_proj, o_proj
```

Gemma-3n uses attention-only targets because its MatFormer/per-layer embedding structure is less standard for PEFT than the dense transformer blocks in Qwen and SmolLM.

## Optional Sweep

If a rerun needs broader search, the smallest useful sweep is:

| Parameter | Values |
| --- | --- |
| LoRA rank | 8, 16, 32 |
| Learning rate | 1e-4, 2e-4, 3e-4 |
| Epochs | 3, 4 |

Primary gates for deciding whether a sweep is needed:

- ESI/SATS accuracy below the target threshold.
- identity refusal rate above the threshold.
- pediatric recall below the deployment gate.
- unstable validation loss.

## Optional Preference Stage

DPO is the preferred Phase 2 alignment method if SFT alone fails behavior gates and preference pairs are available.

Suggested starting configuration:

```yaml
dpo:
  learning_rate: 5.0e-5
  beta: 0.1
  num_train_epochs: 1
  per_device_batch_size: 8
  warmup_ratio: 0.05
```

Preference data can be built from human review or from accepted/rejected candidate responses. It should not be generated blindly from the same model that is being optimized.

## Methods Not Used in Phase 1

| Method | Reason |
| --- | --- |
| Full SFT | unnecessary memory and overfit risk for the dataset size |
| QLoRA | memory was available for BF16 LoRA |
| DoRA | extra training complexity not required for the Phase 1 baseline |
| Prompt/prefix tuning and IA3 | insufficient adaptation capacity for clinical triage behavior |
| Adapter modules | inference latency overhead conflicts with mobile deployment |
| RLHF/PPO | requires substantially more infrastructure and preference data |
| ORPO/KTO/SimPO/IPO | valid alternatives, but DPO has more mature tooling in the current stack |

## References

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DoRA](https://arxiv.org/abs/2402.09353)
- [DPO](https://arxiv.org/abs/2305.18290)
- [TRL](https://huggingface.co/docs/trl/index)
- [PEFT](https://huggingface.co/docs/peft/index)
