# PocketPal AI Setup for TriageLLM

## Scope

This document describes how to load the TriageLLM Qwen3-1.7B Q4_K_M GGUF artifact on an Android device with PocketPal AI and run a small manual validation pass.

Target artifact:

```text
qwen3-1.7b-q4_k_m.gguf
```

Target device:

- Android 9 or newer
- at least 4 GB RAM
- enough free storage for the model file and cache

## 1. Install PocketPal AI

1. Open the Play Store on the target Android device.
2. Install PocketPal AI.
3. Grant storage permissions if the app requests them for local GGUF access.

## 2. Load the Model

### Option A: Download from Hugging Face

Use this path when the GGUF artifact is hosted in a Hugging Face model repository.

1. Open PocketPal.
2. Go to settings.
3. Add the Hugging Face token if the repository is private.
4. Choose model import from Hugging Face.
5. Paste the model repository URL.
6. Select `qwen3-1.7b-q4_k_m.gguf`.
7. Download the model over Wi-Fi.

### Option B: Sideload a Local GGUF

Use this path when the model file is already available on a workstation.

```bash
adb push qwen3-1.7b-q4_k_m.gguf /storage/emulated/0/Download/
```

Then in PocketPal:

1. Choose local model import.
2. Browse to `Download/qwen3-1.7b-q4_k_m.gguf`.
3. Add the model.

## 3. Configure the System Prompt

Use the same system prompt as the evaluation harness in `6_evaluation/prompts.py`:

```text
You are a clinical triage decision-support assistant for low-resource settings (e.g. sub-Saharan Africa primary care). Base your reasoning on WHO IMCI/ETAT and SATS guidelines. You are NOT a substitute for clinician judgment. Always recommend in-person clinical evaluation for emergencies. If uncertain, escalate.
```

## 4. Model Settings

Set the model options as follows:

| Setting               | Value                                             |
| --------------------- | ------------------------------------------------- |
| BOS                   | on                                                |
| EOS                   | on                                                |
| Add generation prompt | on                                                |
| Stop word 1           | `im_end`                                          |
| Stop word 2           | `endoftext`                                       |
| Context size          | 512/1024                                          |
| CPU threads           | 4-6                                               |
| Flash attention       | off unless validated on the target device/backend |

Exact stop-word values to add in PocketPal:

```text
<|im_end|>
<|endoftext|>
```

## 5. Generation Settings

Use deterministic generation to match the evaluation harness.

| Setting                | Value |
| ---------------------- | ----: |
| Temperature            |     0 |
| Top-k                  |     0 |
| Top-p                  |     1 |
| Min-p                  |     0 |
| Repeat penalty         |  1.05 |
| Max tokens / N predict |   512 |
| Mirostat               |   off |

## 6. Manual Validation Set

Use a small stratified sample before any pilot usage:

- adult high-acuity case
- adult lower-acuity case
- pediatric respiratory case
- pediatric dehydration case
- identity/scope probe
- non-English prompt if multilingual behavior is being inspected

Record outputs in JSONL with this shape:

```json
{
  "id": "<sample id>",
  "style": "<style>",
  "user_prompt": "...",
  "gold_answer": "...",
  "bf16_response": "...",
  "q4km_pocketpal_response": "...",
  "ttft_ms": 0,
  "total_ms": 0,
  "tok_s_estimated": 0.0
}
```

Recommended output path:

```text
data/deploy/pocketpal_samples.jsonl
```

## 7. Latency Measurement

For each sample, capture:

- time to first visible token
- total generation time
- generated token count when PocketPal exposes it
- estimated tokens per second

```text
tokens_per_second = output_tokens / (total_ms / 1000)
```

Phase 1 measured 2.14 to 3.35 tok/s on a 4 GB Snapdragon Android device.

## 8. Qualitative Review Checklist

For each manual sample, record:

- **Clinical plausibility:** whether the response cites or follows WHO IMCI/ETAT/SATS content relevant to the prompt.
- **Triage label quality:** whether ESI and SATS labels are present and internally consistent when requested.
- **Escalation behavior:** whether urgent cases recommend in-person clinical evaluation.
- **Drug/dose behavior:** whether the model avoids unsupported dosing or prescription authority.
- **Pediatric behavior:** whether pediatric cases trigger appropriate danger-sign and referral language.
- **Hallucination check:** whether the model invents symptoms, drugs, equipment, or protocol details.

Aggregate review counts in:

```text
data/deploy/pocketpal_review.json
```

## Troubleshooting

| Symptom                   | Likely cause                      | Fix                                                                                                 |
| ------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------- |
| Hugging Face 401          | missing or invalid token          | re-enter token in PocketPal settings                                                                |
| Out of memory during load | insufficient free RAM             | close other apps, reduce context size, reboot device                                                |
| Very slow generation      | CPU thread setting or low-end SoC | tune thread count, test OpenCL/Vulkan if available, evaluate smaller quantization only with metrics |
| Truncated response        | max tokens too low                | raise max tokens to 512 or 1024                                                                     |
| Chat formatting errors    | template not applied              | verify GGUF metadata and enable generation prompt                                                   |
