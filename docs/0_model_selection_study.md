# Base Model Selection Study

## Scope

This study selects candidate base models for supervised fine-tuning and mobile deployment. The target artifact is a GGUF Q4_K_M model that can run offline on an Android device with 4 GB RAM through llama.cpp-compatible runtimes.

Training data:

- `data/augment/train.jsonl`
- 33,575 Hugging Face `messages` records
- 13.2% pediatric/infant-oriented content after augmentation

## Deployment Constraints

| Constraint | Target |
| --- | --- |
| Parameter range | 0.6B to 4B |
| Preferred dense range | 1.5B to 3B |
| Q4_K_M weight footprint | at most 2.5 GB |
| Peak RAM including KV cache | at most 3.5 GB on 4 GB Android |
| Primary language | English |
| Runtime | llama.cpp / GGUF |
| License preference | Apache-2.0 or MIT; more restrictive licenses only if materially better |

The key mobile bottleneck is not only weight size. KV cache memory can exceed weight memory at long context lengths, especially above 3B parameters. For 4 GB Android devices, the practical target is 1.5B to 2B parameters with context limited to 8k to 16k unless KV-cache quantization is enabled.

## Evaluation Criteria

- Mobile fit under Q4_K_M.
- Context length of at least 8k.
- Hugging Face chat-template compatibility.
- llama.cpp/GGUF support.
- General capability from public benchmarks.
- Medical benchmark signal where available.
- Instruction following and low refusal risk on clinical prompts.
- License compatibility with educational and humanitarian use.
- Tooling maturity for LoRA SFT and adapter merge.

## Candidate Landscape

| Family | Candidate | Deployment fit | Notes |
| --- | --- | --- | --- |
| Qwen3 | Qwen3-1.7B | strong | Apache-2.0, 32k context, small footprint, ChatML |
| Qwen3 | Qwen3-4B | borderline | stronger quality, tighter RAM on 4 GB devices |
| Gemma 3n | Gemma-3n-E2B-it | strong | mobile-tuned architecture, Gemma license, non-standard PEFT considerations |
| SmolLM3 | SmolLM3-3B | medium | Apache-2.0, strong instruction following, larger mobile footprint |
| Phi | Phi-4-mini-instruct | borderline | MIT, strong general benchmarks, higher refusal-risk concern from previous Phi family testing |
| Granite | Granite-3.2-2B-Instruct | medium | Apache-2.0, long context, smaller ecosystem |
| Falcon3 | Falcon3-3B-Instruct | medium | custom TII license and smaller ecosystem |
| OLMo-2 | OLMo-2-1B | weak | Apache-2.0 but 4k context is too short for the pipeline |
| MedGemma | MedGemma-4B-it | reference only | medical pretraining but HAI-DEF license and text-only overhead from multimodal design |

## Candidate Shortlist

### Primary: Qwen3-1.7B

Rationale:

- Apache-2.0 license.
- Small Q4_K_M footprint with margin for Android 4 GB.
- 32k context support.
- ChatML compatibility with the Step 3 and Step 4 `messages` data.
- Mature support across transformers, vLLM, PEFT, and llama.cpp.
- Same model family used for preprocessing, reducing tokenizer/template friction.

### Quality Fallback: SmolLM3-3B

Rationale:

- Apache-2.0 license.
- Strong instruction-following signal for sub-4B models.
- Multilingual support including Italian.
- Fits 4 GB targets with tighter memory assumptions and KV-cache control.

Risk:

- Larger mobile footprint and architecture-specific runtime considerations.

### Mobile-Optimized Fallback: Gemma-3n-E2B-it

Rationale:

- Architecture explicitly designed for mobile use.
- Targeted RAM profile around 2 GB to 3 GB depending on configuration.
- Good candidate when runtime latency dominates.

Risk:

- Non-standard MatFormer/per-layer embedding design complicates LoRA target selection.
- Gemma license requires careful review for deployment terms.

## Models Excluded from Phase 1 Deployment

| Model or family | Reason |
| --- | --- |
| Qwen3-8B and larger dense models | Q4_K_M weight and peak RAM exceed the 4 GB Android target |
| Qwen3 MoE variants | file size and MoE routing maturity are unsuitable for the target runtime |
| Llama3-Med42-8B and other 7B+ medical models | mobile footprint exceeds target |
| BioMedLM 2.7B | license restrictions conflict with the project use case |
| OLMo-2-1B | context length too short |
| Apollo-2B | context and domain focus do not match the English triage pipeline |
| Med-V1 / MedS3 | biomedical attribution/curriculum focus rather than triage decision support |

## Training Hardware

The SFT stage targets NVIDIA L40S 46 GB GPUs with BF16 support. LoRA SFT for the shortlisted models fits comfortably on one GPU per model.

Expected stack:

- transformers 4.55+
- TRL 0.13+
- PEFT 0.14+
- accelerate 1.2+
- BF16 training
- gradient checkpointing

## Decision Record

| Slot | Selected model | Role |
| --- | --- | --- |
| A | Qwen3-1.7B | primary candidate |
| B | SmolLM3-3B | quality fallback |
| C | Gemma-3n-E2B-it | mobile-optimized fallback |

The Step 6 evaluation selected Qwen3-1.7B as the Phase 1 deployment model.
