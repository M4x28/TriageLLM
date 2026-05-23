# Step 3: Q&A Rewrite

## Scope

Step 3 converts the cleaned Step 2 records into supervised fine-tuning examples in Hugging Face `messages` format. The stage rewrites MIETIC cases and guideline sections into user/assistant interactions aligned with the deployment scenario.

Input:

- `data/processed/mietic.jsonl`
- `data/processed/sats_sections.jsonl`
- `data/processed/imci_sections.jsonl`
- `data/processed/etat_sections.jsonl`

Output:

- `data/rewrite/train_cases.jsonl`
- `data/rewrite/train_sections.jsonl`
- `data/rewrite/train.jsonl`

## Architecture

```text
3_qa_rewrite/
|-- common.py             # shared helpers and Step 2 vLLM client import
|-- prompts.py            # style prompts and question schema
|-- rewrite_cases.py      # MIETIC cases to Q&A records
|-- rewrite_sections.py   # guideline sections to Q&A records
|-- run_all.py            # stage orchestrator and merge
|-- sample_inspect.py     # spot inspection helper
`-- requirements.txt
```

The stage reuses Step 2 generation utilities through `importlib.util` rather than duplicating the vLLM wrapper.

## Output Schema

```json
{
  "id": "qa_<source_id>_<style>",
  "messages": [
    {"role": "system", "content": "<triage assistant system prompt>"},
    {"role": "user", "content": "<generated question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "metadata": {
    "source_id": "mietic_000000",
    "source": "mietic | sats | who_imci | who_etat",
    "doc_type": "case | guideline_section",
    "style": "<style>",
    "task_type": "esi1_detection | esi2_detection | resource_prediction | other | null",
    "prompt_version": "v1"
  }
}
```

`messages` is the OpenAI/Hugging Face chat format consumed by TRL `SFTTrainer`.

## Rewrite Styles

### MIETIC Cases

| Style | Intended voice |
| --- | --- |
| `caregiver_query` | family member or patient using plain language |
| `clinician_handoff` | triage nurse or emergency clinician using clinical shorthand |
| `field_worker_query` | health worker in a low-resource facility |

### Guideline Sections

| Style | Focus |
| --- | --- |
| `direct_question` | direct clinical question |
| `protocol_lookup` | explicit request for guideline content |
| `red_flag_check` | escalation, referral, and danger-sign checks |

## Case Answer Construction

For MIETIC cases, the assistant answer combines the original reasoning trace with an explicit triage recommendation when the source task supports one.

```text
<reasoning_trace>

---
Triage recommendation: ESI <level> / SATS <color>
```

Deterministic mapping:

| MIETIC task type | Added label |
| --- | --- |
| `esi1_detection` | ESI 1 / SATS Red |
| `esi2_detection` | ESI 2 / SATS Orange |
| `resource_prediction` | no added ESI/SATS label |
| `other` | no added ESI/SATS label |

For guideline sections, the assistant answer is the section text without synthetic clinical conclusions.

## Generation Strategy

The stage uses one LLM call per `(record, style)` pair. This isolates failures by style and gives each generated question its full attention budget. Guided JSON decoding enforces syntax for generated question records.

## Section Skip Filter

The LLM can mark a generated section record as skipped when the source chunk is table of contents material, acknowledgements, preface text, references, page footers, or other non-clinical content.

## Execution

```bash
pip install -r 3_qa_rewrite/requirements.txt
python 3_qa_rewrite/run_all.py
```

## Completion Checks

- `data/rewrite/train_cases.jsonl` contains the MIETIC-derived Q&A records.
- `data/rewrite/train_sections.jsonl` contains guideline-derived Q&A records.
- `data/rewrite/train.jsonl` contains the merged SFT-ready dataset.
- Every row has exactly three chat messages: system, user, assistant.
- `metadata.style` is one of the six declared rewrite styles.
