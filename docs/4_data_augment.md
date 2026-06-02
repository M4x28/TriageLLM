# Step 4: Data Augmentation

## Scope

Step 4 prepares the final SFT dataset from the Step 3 rewrite output. It filters length outliers, creates a stratified train/validation split, oversamples rare but important slices, adds identity/safety examples, and shuffles deterministically.

Input:

- `data/rewrite/train.jsonl`

Output:

- `data/augment/train.jsonl`
- `data/augment/validation.jsonl`
- `data/augment/augment_stats.json`

## Input Composition

The Step 3 dataset is dominated by MIETIC adult emergency cases.

| Slice             |  Count | Share |
| ----------------- | -----: | ----: |
| MIETIC cases      | 28,886 | 98.1% |
| WHO IMCI sections |    219 | 0.74% |
| WHO ETAT sections |    195 | 0.66% |
| SATS sections     |    147 | 0.50% |

This imbalance is intentional in the source data but unsuitable for the target behavior: the model must retain triage labels while also learning pediatric protocols and safety boundaries.

## Oversampling Policy

Oversampling is applied after the train/validation split. Validation receives only original records.

| Slice                                       | Factor | Purpose                                                             |
| ------------------------------------------- | -----: | ------------------------------------------------------------------- |
| Pediatric guidelines: WHO IMCI and WHO ETAT |    10x | increase pediatric protocol frequency                               |
| Adult guideline: SATS                       |     5x | reinforce SATS color and TEWS language                              |
| MIETIC cases                                |     1x | preserve dominant labeled triage source without additional copies   |
| Identity records                            |    10x | reinforce scope, disclaimer, escalation, and non-authority behavior |

## Identity Records

Identity records cover three personas and six safety patterns.

Personas:

- `caregiver_query`
- `clinician_handoff`
- `field_worker_query`

Patterns:

- `identity_request`
- `diagnosis_request`
- `dosing_request`
- `definitive_answer`
- `scope_creep`
- `prescription_request`

The answer templates are fixed. Only the user question variants are paraphrased, which avoids introducing unsupported medical claims into the safety responses.

## Length Filtering

Records with assistant content longer than 4,000 characters are dropped. The stage does not truncate records because partial assistant messages can train malformed outputs.

## Train/Validation Split

The split is stratified on:

```text
(doc_type, style, source)
```

The split ratio is 95/5. Strata with fewer than 20 records are kept out of validation to avoid singleton leakage.

## Pediatric Emphasis

Pediatric/infant-oriented content increases from 1.5% before augmentation to 13.2% after augmentation.

| Slice                           | Before | After oversampling |
| ------------------------------- | -----: | -----------------: |
| WHO IMCI                        |    219 |              2,070 |
| WHO ETAT                        |    195 |              1,860 |
| Identity caregiver records      |     54 |                510 |
| Total pediatric/infant-oriented |    468 |              4,440 |

## Output Schema Additions

Step 4 preserves the Step 3 `messages` schema and adds split/oversampling metadata.

```jsonc
{
  "id": "qa_mietic_000000_caregiver_query__x3",
  "messages": [],
  "metadata": {
    "source": "mietic | who_imci | who_etat | sats | identity",
    "doc_type": "case | guideline_section | identity",
    "style": "caregiver_query | clinician_handoff | field_worker_query | direct_question | protocol_lookup | red_flag_check",
    "task_type": "esi1_detection | esi2_detection | resource_prediction | other | null",
    "oversample_tag": "original | copy_2 | ... | copy_10",
    "split": "train | validation",
    "pattern": "<identity pattern when applicable>"
  }
}
```

## Final Counts

| Artifact                          |         Records |
| --------------------------------- | --------------: |
| `data/augment/train.jsonl`        |          33,575 |
| `data/augment/validation.jsonl`   |           1,479 |
| `data/augment/augment_stats.json` | sidecar metrics |

Training composition:

- MIETIC: 81.6%
- pediatric guidelines: 11.7%
- adult guidelines: 2.1%
- identity records: 4.6%

## Execution

```bash
python 4_data_augment/run_all.py
```

Identity seed generation is available as a separate script when regeneration is required:

```bash
python 4_data_augment/generate_identity_seed.py \
  --model Qwen/Qwen3-32B \
  --variants 8 \
  --out 4_data_augment/identity_seed.jsonl
```

## Completion Checks

- `train.jsonl` has 33,575 records.
- `validation.jsonl` has 1,479 records.
- Validation contains only records with `oversample_tag = "original"`.
- Every record has system, user, and assistant messages.
- Every record has `metadata.split` and `metadata.oversample_tag`.

---

## Phase 2 Addendum: Label-First Reformatting

This section was added during Phase 2 after a failure mode surfaced only when
scaling to large models. It documents a change to the assistant-message
formatting that the original Phase 1 augmentation did not require.

### Problem observed in Phase 2

The Phase 1 assistant messages place the triage label at the *end* of a long
clinical narrative (the analysis is written first, the `ESI N` / `SATS Color`
conclusion comes last). On the Phase 1 primary (Qwen3-1.7B) this was harmless:
internal eval reached ESI 0.928 / SATS 0.928, because a 1.5B model produces
short, template-like answers and emits the label before exhausting the
generation budget.

On the Phase 2 large tier the same data taught the opposite behavior. The 27B
models inherited a strong prior for verbose clinical prose, so they spend the
entire token budget on analysis and the label is either truncated or never
reached:

| Model                | max_new_tokens | ESI accuracy | Label present rate |
| -------------------- | -------------: | -----------: | -----------------: |
| Qwen3-1.7B (Phase 1) |           1024 |        0.928 | n/a (not measured) |
| medgemma-27b         |            256 |        0.008 |               0.8% |
| medgemma-27b         |            512 |        0.720 |                72% |
| qwen3.6-27b          |            256 |        0.036 |               2.0% |
| qwen3.6-27b          |            512 |        0.660 |                66% |

Even at 512 tokens, 28 to 34 percent of large-model responses contain no
parseable label. For a triage system this is a deployment blocker: if the
downstream parser cannot find `ESI N` or `SATS Color`, the UI cannot render the
triage colour to the health worker, which is the primary function of the
system.

The failure is proportional to model size: the larger the model, the stronger
the pull toward long-form narrative before the label. The fix must therefore be
applied at the data level, independent of the generation budget.

### Reformatting rule

For every record whose gold assistant message contains a triage label, prepend
the label as the first line, before any analysis:

```text
**Triage: ESI 2 / SATS Orange**

<original analysis text>
```

Rules:

- Only records with a parseable gold label are modified (~70% of the corpus,
  the MIETIC-derived labeled cases). Guideline sections (WHO IMCI / ETAT / SATS)
  and identity records are left unchanged.
- When only one label system is present in the gold answer, emit only that one
  (`**Triage: ESI 2**` or `**Triage: SATS Orange**`).
- The original analysis body is preserved verbatim after the header, so no
  clinical content is lost or paraphrased.

This places the label in the first 10 to 20 tokens, making label presence
independent of response length or `max_new_tokens`.

### Implementation

The reformatting is integrated into the existing Phase 4 pipeline rather than a
standalone script. `prepend_triage_label()` lives in
`4_data_augment/common.py` and is called by `4_data_augment/augment.py`
immediately after loading the rewrite output and before length filtering, so
the label header is in place before the train/validation split and
oversampling propagate it to every copy.

The label parser reuses the exact ESI/SATS regex from
`6_evaluation/prompts.py`, so the prepended header matches what the evaluator
extracts from the gold answer. The function is idempotent: an assistant message
already starting with `**Triage:` is skipped.

No separate backup is written. The pipeline input `data/rewrite/train.jsonl`
is immutable, so the labeled dataset is regenerated deterministically by
re-running the augmentation step.

### Execution

```bash
python 4_data_augment/run_all.py
```

### Completion checks (Phase 2)

- Every record with a gold triage label begins its assistant message with a
  `**Triage: ...**` header.
- Guideline and identity records have no header.
- Record counts match the Phase 1 totals (33,575 train / 1,479 validation).