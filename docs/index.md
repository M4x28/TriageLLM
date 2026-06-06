# TriageLLM Documentation Index

## Macro Pipeline Steps

1. **Data fetch**: download shortlisted sources (MIETIC + SATS + WHO IMCI/ETAT).
2. **Data cleaning**: regex + LLM filter, dedup, langid, length.
3. **Q&A rewrite**: monologues/sections -> question/answer pairs.
4. **Data augmentation**: stylistic variants, same answer.
5. **SFT training**: LoRA on 3 parallel candidates, merge -> HF checkpoint.
6. **Evaluation**: post-eval on validation + identity refusal + triage F1, primary model decision.
7. **Deploy**: GGUF Q4_K_M + Android test via llama.cpp.

## Phase 1 Documents

| #   | Doc                                                                             | Covered step |
| --- | ------------------------------------------------------------------------------- | ------------ |
| 0   | [Data source study](0_data_sources_study.md)                                    | pre-step 1   |
| 0   | [Deployment model selection study](0_model_selection_study.md)                  | pre-step 5   |
| 0   | [Preprocessing model selection study](0_preprocessing_model_selection_study.md) | pre-step 2   |
| 0   | [LLM quantization study](0_quantization_study.md)                               | pre-step 7   |
| 0   | [Fine-tuning techniques study](0_hyperparameters_search_study.md)               | pre-step 5   |
| 1   | [Data fetch: design choices](1_data_fetch.md)                                   | step 1       |
| 2   | [Data cleaning: design choices](2_data_cleaning.md)                             | step 2       |
| 3   | [Q&A rewrite: design choices](3_qa_rewrite.md)                                  | step 3       |
| 4   | [Data augmentation: design choices](4_data_augment.md)                          | step 4       |
| 5   | [SFT training: design choices](5_sft_training.md)                               | step 5       |
| 6   | [Evaluation: design choices](6_evaluation.md)                                   | step 6       |
| 7   | [GGUF Q4_K_M deploy + PocketPal Android](7_deploy.md)                           | step 7       |

## Phase 2 Documents

Advanced evaluation: build a robust, reproducible behavioral eval before scaling models.

| #   | Doc                                                                       | Covered step      |
| --- | ------------------------------------------------------------------------- | ----------------- |
| 8   | [Small-model re-SFT + corrected-gold eval](6_evaluation.md)               | step 8            |
| 0   | [Bloom evaluation study (behavioral)](0_phase2_bloom_evaluation_study.md) | pre-step 8 (eval) |
| 6b  | [Petri behavior exploration](6b_petri_behavior_exploration.md)            | step 6b (eval)    |

### Phase 2 data-pipeline fixes

| Fix | What                                            | Step | Doc                                    |
| --- | ----------------------------------------------- | ---- | -------------------------------------- |
| F1  | ESI label from the task answer, not `task_type` | 3    | [3_qa_rewrite.md](3_qa_rewrite.md)     |
| F2  | Label-first: `ESI/SATS` on the first line       | 4    | [4_data_augment.md](4_data_augment.md) |

## Phase 3 Documents

Model scaling and large-model fine-tuning.

| #   | Doc                                                                    | Covered step |
| --- | ---------------------------------------------------------------------- | ------------ |
| 0   | [Model selection study (dual tier)](0_phase3_model_selection_study.md) | pre-step 9   |
| 0   | [Quantization study (mid + high tier)](0_phase3_quantization_study.md) | pre-step 9   |


## Lessons Learned

### 1. Oversized preprocessing model (Steps 2-4)

For structured MIETIC extraction + identity paraphrasing + Q&A rewrite we deployed
**Qwen3-32B** BF16 (62 GB) on an A100 80 GB. Actual ETA: 1.5 h. **Qwen3-8B** (16 GB) would have been sufficient, with an expected 6 to 10x speedup.
Estimated saving: 4 to 6 server hours. Rule: for ETL tasks, start from
the smallest plausible model, validate on 100 records, and scale only when there is evidence of failure modes. Details: [`0_preprocessing_model_selection_study.md`](0_preprocessing_model_selection_study.md).

### 2. Q4_K_M beats Q5_K_M on the primary task (Step 7)

Three-way ablation BF16 vs Q4_K_M vs Q5_K_M on a 400-record validation set with
greedy decoding (temp 0): Q4_K_M (5.12 bpw, 1.1 GB) reaches ESI 0.920 vs Q5_K_M
(5.82 bpw, 1.2 GB) ESI 0.844. This is an 8.4-point drop even though Q5 is more precise.
Probable cause: different weight outlier preservation across K-quant schemes. Q4_K_M instead loses
on pediatric long-form content (semantic hallucination). Rule: do not assume
more bits = higher accuracy. Always ablate adjacent K-quants before deployment.
Details in Sections 7.2-7.3 of [`7_deploy.md`](7_deploy.md).

### 3. NLL through llama-cpp-python is not comparable with HF (Step 7)

The `logprobs` API in llama-cpp-python on Qwen3 returns inconsistent values
(NLL=25, PPL=10^11) because of a BPE tokenizer mismatch when the HF chat template
is applied and retokenized on the GGUF side. NLL is not a reliable post-quant metric
with this backend. Workaround: skip NLL as a gate and use only ESI/SATS/refusal/pediatric.
For Phase 2, evaluate the native llama.cpp backend with `--logits` plus parsing tools/perplexity.
Details in Section 7.4 of [`7_deploy.md`](7_deploy.md).

### 4. Eval a 27B: single-GPU 4-bit beats BF16 multi-GPU

With `device_map="auto"` across 2 GPUs (pipeline parallelism over PCIe),
batch-1 generation drops to about 3 tok/s for a 27B BF16 model: every token
transfers activations between GPUs at each layer. Result: 255 samples x ~130 s
each = about 9 h of eval for a single model.

Fix: load the merged checkpoint in BnB 4-bit on a single GPU (27B -> ~14 GB,
fits one L40S 46 GB), or use vLLM tensor parallelism. Expected: 10-20 tok/s,
eval in about 1 h. Measured: probes that took 2h42m on multi-GPU BF16 finished
in 1 minute on single-GPU 4-bit (160x).

Operating rules for Phase 2 onward:
- `--n-ppl-samples 0` for Qwen3.6 models (CUDA assert on NLL with multi-GPU,
  not recoverable in-process)
- `--max-new-tokens 512`: a smaller budget truncates the answer before the
  model reaches the triage label (256 collapsed label presence to under 2%)
- load eval on a single GPU in BnB 4-bit, not BF16 multi-GPU

### 5. A degenerate gold label makes accuracy meaningless (Phase 2)

Step-3 `derive_triage_label` mapped MIETIC `task_type` straight to ESI
(`esi1_detection` -> ESI 1), but the tasks are ESI decision points, not patient
levels. Every labeled case collapsed to `ESI 1 / SATS Red` in train and
validation, making the Phase 1 0.928 meaningless (the model only echoed a
constant) and penalising the larger Phase 2 models that reasoned correctly.
Tell-tale sign: ESI accuracy exactly equalled SATS accuracy (232/250 both),
impossible on a real label space. Fix: derive the level from the task answer
(life-saving -> ESI 1, resources -> ESI 3/4/5, danger-zone vitals -> ESI 2).
Rule: inspect the gold distribution before trusting accuracy; independent
metrics reading identical signal a collapsed label space. Details: [3_qa_rewrite.md](3_qa_rewrite.md).