# Step 7: Deployment

## Scope

Step 7 validates the selected model after quantization and documents the Android deployment path. The Phase 1 primary model is `qwen3-1.7b`, selected in Step 6 with a composite score of 0.9285.

Input:

- `data/sft/checkpoints/qwen3-1.7b/merged/`
- GGUF artifact produced with llama.cpp conversion and quantization tools
- Step 6 BF16 evaluation JSON

Output:

- `data/deploy/gguf/qwen3-1.7b-q4_k_m.gguf`
- `data/deploy/eval_q4km/qwen3-1.7b.json`
- `data/deploy/compare_bf16_q4km.json`
- private Hugging Face model repository
- Android/PocketPal validation notes

## Architecture

```text
7_deploy/
|-- common.py              # Step 6 imports, GGUF loading, generation helpers
|-- eval_gguf.py           # llama-cpp-python evaluation on validation records
|-- compare.py             # BF16 vs Q4_K_M metric comparison
|-- emergent_probes.py     # out-of-distribution skill probes
|-- inspect_pediatric.py   # pediatric-output inspection helper
|-- publish_to_hf.py       # private HF upload and model-card generation
|-- pocketpal_setup.md     # Android runtime setup
`-- requirements.txt
```

The repository contains evaluation and publishing utilities. GGUF conversion itself is performed with llama.cpp tools against the merged Hugging Face checkpoint.

## Quantization Pipeline

```text
merged HF checkpoint
  -> llama.cpp convert_hf_to_gguf.py
  -> F16 GGUF
  -> llama-quantize Q4_K_M
  -> qwen3-1.7b-q4_k_m.gguf
```

Q4_K_M is a grouped 4-bit GGUF quantization scheme with double-quantized scales. It is calibration-free and supported by llama.cpp runtimes on CPU, CUDA, Metal, and Android wrappers.

## Post-Quantization Evaluation

`eval_gguf.py` mirrors the Step 6 internal evaluation with the inference backend changed to `llama-cpp-python`.

```bash
python 7_deploy/eval_gguf.py \
  --model qwen3-1.7b \
  --gguf data/deploy/gguf/qwen3-1.7b-q4_k_m.gguf \
  --n-ppl-samples 0
```

`--n-ppl-samples 0` is used in Phase 1 because Qwen3 log-probability evaluation through llama-cpp-python produced inconsistent NLL values when compared with the Hugging Face tokenizer path.

Compare BF16 and Q4_K_M metrics:

```bash
python 7_deploy/compare.py \
  --bf16 data/eval/internal/qwen3-1.7b.json \
  --q4km data/deploy/eval_q4km/qwen3-1.7b.json
```

## Publishing

The GGUF is published to a private Hugging Face model repository:

```bash
python 7_deploy/publish_to_hf.py \
  --gguf data/deploy/gguf/qwen3-1.7b-q4_k_m.gguf \
  --repo <username>/triagellm-qwen3-1.7b-gguf \
  --private
```

The publisher uploads:

- the GGUF artifact
- the generated model card
- SHA-256 checksum
- BF16 evaluation JSON
- Q4_K_M evaluation JSON
- BF16-vs-Q4_K_M comparison JSON

## PocketPal Android Configuration

PocketPal AI is a llama.cpp-based Android runtime. The detailed setup is in [PocketPal setup](../7_deploy/pocketpal_setup.md).

Required model settings:

- BOS: on
- EOS: on
- Add generation prompt: on
- System prompt: `SYSTEM_PROMPT` from `6_evaluation/prompts.py`
- Stop words: `<|im_end|>` and `<|endoftext|>`

Recommended generation settings:

- temperature: `0`
- top-k: `0`
- top-p: `1`
- min-p: `0`
- repeat penalty: `1.05`
- max tokens: `512`
- context size: `2048` or `4096` for longer prompts

## Results

### Quantized Artifacts

| File | Effective bpw | Size | Notes |
| --- | ---: | ---: | --- |
| F16 GGUF | 16.00 | 3.3 GB | intermediate |
| Q4_K_M GGUF | 5.12 | 1.1 GB | selected deployment artifact |
| Q5_K_M GGUF | 5.82 | 1.2 GB | ablation artifact |

### BF16 vs Quantized Validation

| Metric | BF16 GPU | Q4_K_M | Q5_K_M |
| --- | ---: | ---: | ---: |
| ESI accuracy | 0.928 | 0.920 | 0.844 |
| SATS accuracy | 0.928 | 0.920 | 0.844 |
| Pediatric recall | 0.787 | 0.437 | 0.508 |
| Refusal rate | 0.000 | 0.000 | 0.000 |
| Server throughput | 59.9 tok/s GPU | 27.92 tok/s CPU | 27.57 tok/s CPU |

Q4_K_M retained the primary triage-label metrics better than Q5_K_M in this ablation despite the lower bit rate.

### Android Device Validation

Device test: Snapdragon Android phone with 4 GB RAM, PocketPal CPU threads set to 5.

| Scenario | Result | Throughput |
| --- | --- | ---: |
| Adult ACS-style case | ESI 1 / SATS Red parsed correctly | 2.14 tok/s |
| Pediatric severe-pneumonia-style case | no hallucinated content, but weaker label formatting | 2.57 tok/s |
| French pediatric prompt | French comprehension with English response; mistranslation observed | 3.35 tok/s |

PocketPal throughput is below the Phase 1 target of 8 tok/s.

## Known Limitations

- Pediatric validation in Phase 1 has only five IMCI/ETAT records in the sampled quantized evaluation subset.
- Q4_K_M degrades pediatric long-form guideline recall.
- llama-cpp-python NLL/log-probability values were not reliable for Qwen3 in the Phase 1 setup.
- Android throughput on the tested 4 GB device is below target.
- Phase 1 SFT is English-only; multilingual prompts require Phase 2 data and evaluation.
