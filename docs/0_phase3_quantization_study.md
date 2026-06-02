# Phase 3 Quantization Study: mid and high tiers

## 1. Scope

Quantization is chosen per tier. The mobile **mid** tier must hit a small
footprint, so Q4_K_M is mandatory, now applied to 4-8B models. The server
**high** tier has no footprint limit: the axis shifts from "smallest that fits" to "best accuracy/throughput trade-off for the latency target". This doc guides the scheme each tier's primary adopts at deploy.

## 2. High-tier constraints

- Inference on one L40S, batch 1.
- Throughput target: a ~200-token response in <= 5 s (>= 40 tok/s). A 27B dense
  in BF16 sustains ~25-35 tok/s; Q4_K_M reaches ~50-60 tok/s.
- Accuracy budget: <= 1-point drop on ESI/SATS vs the post-SFT BF16 primary;
  format compliance unchanged.

## 3. Scheme comparison (27B dense reference)

| Scheme     | bpw | Size 27B | Speedup vs BF16 | Accuracy drop (PPL) |
| ---------- | --- | -------- | --------------- | ------------------- |
| BF16       | 16  | ~55 GB   | 1.0x            | 0                   |
| FP8 (E4M3) | 8   | ~28 GB   | 1.5-2.0x        | ~0                  |
| Q8_0       | 8.5 | ~30 GB   | 1.4x            | ~0                  |
| Q6_K       | 6.2 | ~22 GB   | 1.6x            | < 0.01              |
| Q5_K_M     | 5.5 | ~19 GB   | 1.7x            | ~0.026              |
| Q4_K_M     | 4.5 | ~15 GB   | 1.9x            | +0.054              |
| AWQ INT4   | 4   | ~14 GB   | 2.0x (vLLM)     | ~0.04               |

FP8 and AWQ open GPU-server inference paths (vLLM, TensorRT-LLM) that were unavailable in the mobile target.

## 4. Mid-tier scheme

Fixed at Q4_K_M: the only scheme that fits a 4-8B model into a mid-range phone's
RAM while preserving triage accuracy. Phase 1 even showed Q4_K_M can beat Q5_K_M
on the primary task. Post-SFT, each mid candidate is re-evaluated after Q4_K_M
conversion before the mobile primary is chosen.

## 5. High-tier ablation

The table gives a range; the final choice comes from an empirical ablation on
the post-SFT high-tier primary:

1. Export to BF16, Q6_K, Q5_K_M, Q4_K_M (AWQ/FP8 optional later).
2. Run the full eval battery per scheme: ESI/SATS accuracy, format compliance,
   pediatric recall, emergent probes.
3. Pick the scheme with the smallest ESI/SATS drop and format compliance >= 0.95.

## 6. Runtime

- BF16 / FP8 / AWQ: vLLM (paged attention).
- K-quants (Q4_K_M / Q5_K_M / Q6_K): llama.cpp server mode.

The winning scheme decides the runtime.

## 7. Preliminary decision

Pre-ablation default for the high tier: **Q5_K_M** the literature sweet spot
between near-lossless accuracy and ~1.7x throughput for 13-70B models on server
CPU SIMD. Drop to Q4_K_M only if the ablation shows negligible loss on the
triage task. Q2_K and Q3_K_M are excluded a priori (~9-pt reasoning drop,
clinically unacceptable). The mid tier stays Q4_K_M.

## 8. References

- [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [vLLM AWQ guide](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)
- [SLMQuant 2025](https://arxiv.org/pdf/2511.13023)
- [Model selection study](0_phase3_model_selection_study.md)
