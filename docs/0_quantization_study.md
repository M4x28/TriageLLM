# Quantization Study

## Scope

This study defines the quantization strategy for deploying TriageLLM on Android devices with limited RAM. The deployment target is a llama.cpp-compatible GGUF file.

## Quantization Basics

Quantization reduces the numerical precision of model weights, and sometimes activations, from floating point to lower-bit representations. The goal is lower memory use, lower bandwidth pressure, and faster inference while preserving enough task accuracy.

Affine quantization maps a floating-point value to an integer bucket:

```text
q = round(x / scale) + zero_point
x_reconstructed = (q - zero_point) * scale
```

Important design choices:

- **Per-tensor quantization:** one scale for the whole tensor; efficient but weak with outliers.
- **Per-channel quantization:** one scale per output channel; better weight fidelity.
- **Per-group quantization:** scales for small blocks of weights; common for modern 4-bit LLM quantization.
- **Calibration-based PTQ:** uses representative examples to tune scales, as in GPTQ or AWQ.
- **Calibration-free PTQ:** uses only model weights, as in many GGUF K-quant flows.
- **QAT:** trains or adapts the model while simulating quantization noise.

## Technique Summary

| Technique     | Type                        | Strengths                                        | Limitations                                       |
| ------------- | --------------------------- | ------------------------------------------------ | ------------------------------------------------- |
| GGUF K-quants | calibration-free PTQ        | native llama.cpp/mobile support, fast conversion | weight-only; quality depends on quant choice      |
| GPTQ          | calibration PTQ             | strong CUDA server inference ecosystem           | less suitable for Android CPU deployment          |
| AWQ           | calibration PTQ             | protects salient channels, strong INT4 retention | needs calibration set and GPU-oriented deployment |
| SmoothQuant   | W8A8 PTQ                    | stable server INT8 path                          | not a mobile INT4 solution                        |
| HQQ           | calibration-free PTQ        | fast iteration and competitive INT4 retention    | smaller mobile ecosystem than GGUF                |
| NF4 / QLoRA   | training-time quantization  | excellent for memory-efficient fine-tuning       | not the preferred final mobile inference format   |
| EXL2          | mixed-bit PTQ               | high CUDA throughput                             | CUDA-only                                         |
| FP8           | FP8 inference               | datacenter acceleration on supported hardware    | irrelevant for Android deployment                 |
| W4A4 QAT      | quantization-aware training | maximum compression                              | extra training cost and higher accuracy risk      |

## GGUF K-Quant Options

| Variant | Effective bits/weight | Typical use                            |
| ------- | --------------------: | -------------------------------------- |
| Q2_K    |             about 2.6 | extreme compression, high quality loss |
| Q3_K_M  |             about 3.4 | RAM-constrained experiments            |
| Q4_K_M  |             about 4.5 | default mobile balance                 |
| Q5_K_M  |             about 5.5 | higher precision with larger file      |
| Q6_K    |             about 6.2 | quality-focused with larger RAM budget |
| Q8_0    |             about 8.5 | near-lossless reference                |

For TriageLLM, the task is label-sensitive and safety-sensitive. Sub-4-bit options are too risky for the Phase 1 deployment target.

## Mobile Memory Budget

Target device: Android phone with 4 GB RAM.

Approximate budget:

- Android OS and background processes: 0.5 GB to 1.0 GB.
- Model weights plus KV cache: about 2.0 GB to 2.5 GB preferred.
- Context size: 2k to 8k practical for Phase 1.

| Candidate       | Q4_K_M weights | KV cache at 8k | Expected peak |
| --------------- | -------------: | -------------: | ------------: |
| Qwen3-1.7B      |   about 0.8 GB |   about 0.4 GB |  about 1.4 GB |
| SmolLM3-3B      |   about 1.5 GB |   about 0.7 GB |  about 2.4 GB |
| Gemma-3n-E2B-it |   about 1.5 GB |   about 0.6 GB |  about 2.3 GB |

## Selected Strategy

Phase 1 uses **GGUF Q4_K_M**.

Rationale:

- native llama.cpp support
- Android runtime compatibility
- no calibration dataset required
- small enough for a 4 GB device
- materially better accuracy retention than Q3-style compression
- easy evaluation through `llama-cpp-python`

## Alternatives

- **Q5_K_M:** evaluated as a higher-bit ablation; did not improve primary ESI/SATS accuracy in Phase 1.
- **Q3_K_M:** potential future latency experiment for very slow phones, but expected to degrade reasoning and label reliability.
- **AWQ/GPTQ:** useful for GPU-serving variants, not the mobile target.
- **QAT:** reserved for future work if PTQ fails strict deployment gates.

## Deployment Pipeline

```text
LoRA SFT checkpoint
  -> merge adapter into base model
  -> save Hugging Face safetensors
  -> convert to F16 GGUF with llama.cpp
  -> quantize to Q4_K_M
  -> evaluate with llama-cpp-python
  -> publish private GGUF repository
```