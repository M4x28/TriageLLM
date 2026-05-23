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

| Slice | Count | Share |
| --- | ---: | ---: |
| MIETIC cases | 28,886 | 98.1% |
| WHO IMCI sections | 219 | 0.74% |
| WHO ETAT sections | 195 | 0.66% |
| SATS sections | 147 | 0.50% |

This imbalance is intentional in the source data but unsuitable for the target behavior: the model must retain triage labels while also learning pediatric protocols and safety boundaries.

## Oversampling Policy

Oversampling is applied after the train/validation split. Validation receives only original records.

| Slice | Factor | Purpose |
| --- | ---: | --- |
| Pediatric guidelines: WHO IMCI and WHO ETAT | 10x | increase pediatric protocol frequency |
| Adult guideline: SATS | 5x | reinforce SATS color and TEWS language |
| MIETIC cases | 1x | preserve dominant labeled triage source without additional copies |
| Identity records | 10x | reinforce scope, disclaimer, escalation, and non-authority behavior |

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

| Slice | Before | After oversampling |
| --- | ---: | ---: |
| WHO IMCI | 219 | 2,070 |
| WHO ETAT | 195 | 1,860 |
| Identity caregiver records | 54 | 510 |
| Total pediatric/infant-oriented | 468 | 4,440 |

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

| Artifact | Records |
| --- | ---: |
| `data/augment/train.jsonl` | 33,575 |
| `data/augment/validation.jsonl` | 1,479 |
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
