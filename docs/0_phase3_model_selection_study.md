# Phase 3 Model Selection Study: dual-tier scaling (mobile + server)

## 1. Scope

Phase 3 scales beyond the Phase 1 primary (Qwen3-1.7B) to test three goals
raised in academic review:

1. **Performance**: whether larger models capture the WHO IMCI/ETAT/SATS triage
   domain better.
2. **Emergent abilities**: format adherence, dose math, differential ranking and
   theory of mind all failed in Phase 1; a scale jump should activate them.
3. **Multilingual**: French is optional but wanted for a future francophone
   Africa rollout.

Architecture is dual-tier: a mobile **mid** tier (mid-range phones) and an
unconstrained server **high** tier. The existing training, evaluation and deploy
pipeline is reused unchanged.

## 2. Selection criteria

| Criterion        | Mid tier (mobile)            | High tier (server)            |
| ---------------- | ---------------------------- | ----------------------------- |
| Parameters       | 3.8-8 B                      | >= 27 B (dense or MoE)        |
| Q4_K_M footprint | <= 4.5 GB                    | irrelevant                    |
| License          | Apache-2.0 / MIT preferred   | Apache-2.0 / MIT / Gemma ok   |
| Ecosystem        | native llama.cpp + GGUF      | llama.cpp or vLLM             |
| ARM Neon         | required                     | not required                  |
| Training         | LoRA r=16 on one L40S        | LoRA r=8/r=4 + grad ckpt      |
| Medical pretrain | bonus                        | bonus                         |

## 3. Mid-tier shortlist (mobile, 3)

| Slot | Model               | Params | License    | HF id                           | Rationale                                                                 |
| ---- | ------------------- | ------ | ---------- | ------------------------------- | ------------------------------------------------------------------------- |
| M1   | Qwen3-8B            | 8 B    | Apache-2.0 | `Qwen/Qwen3-8B`                 | 4.7x jump over Phase 1; same family/ChatML, dataset already compatible.   |
| M2   | Phi-4-mini-instruct | 3.8 B  | MIT        | `microsoft/Phi-4-mini-instruct` | 22 native languages (FR/AR/PT/IT); tests multilingual without an FR set.  |
| M3   | MedGemma-1.5-4B-it  | 4 B    | Gemma      | `google/medgemma-1.5-4b-it`     | Medical-pretrained SOTA (91% MedQA); optional text+image.                 |

## 4. High-tier shortlist (server, 3)

| Slot | Model                | Params               | License    | HF id                         | Rationale                                                          |
| ---- | -------------------- | -------------------- | ---------- | ----------------------------- | ----------------------------------------------------------------- |
| L1   | Qwen3.6-27B          | 27 B dense           | Apache-2.0 | `Qwen/Qwen3.6-27B`            | Apr 2026 flagship dense; multimodal; strong reasoning.            |
| L2   | Qwen3.6-35B-A3B      | 35 B / 3 B active MoE| Apache-2.0 | `Qwen/Qwen3.6-35B-A3B`        | MoE: ~4B-class throughput with 35B capacity; efficiency vs quality.|
| L3   | MedGemma-27B-text-it | 27 B text            | Gemma      | `google/medgemma-27b-text-it` | Medical reference upper bound; 91% MedQA gold standard.           |

## 5. Rejected

- **Qwen3.5-4B**: redundant with Qwen3-8B; the 8B is the more informative jump.
- **Gemma-3-4B**: outclassed by MedGemma-1.5-4B on the medical task.
- **Apollo2-7B**: below MedGemma-1.5-4B on MedQA.
- **Mistral-7B-v0.3**: outclassed by Phi-4-mini on multilingual at smaller size.
- **Qwen3-14B/32B (2025)**: superseded by Qwen3.6-27B.
- **MedGemma-27B v1**: superseded by 1.5.
- **DeepSeek V4 Flash**: ~570 GB BF16, too large to SFT on a single GPU.
- **Llama 4**: Meta Custom license (700M MAU clause), risky for an NGO partner.

## 6. SFT strategy

No 0-shot pre-filter: all six candidates go to SFT. The choice is data-driven on
post-SFT results (composite score + format compliance + emergent probes, plus
Q4_K_M post-quant for the mid tier). Each model fits a single L40S: mid tier at
LoRA r=16, high tier at LoRA r=8 with gradient checkpointing.

## 7. References

- [Qwen3.6 collection](https://huggingface.co/collections/Qwen/qwen36)
- [Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35)
- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [Quantization study](0_phase3_quantization_study.md)
