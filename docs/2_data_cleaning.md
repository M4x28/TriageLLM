# Step 2: Data Cleaning

## Scope

Step 2 transforms raw artifacts from Step 1 into unified JSONL records ready for Q&A rewrite.

Operations:

- Parse MIETIC narratives with deterministic regex and LLM-assisted structured extraction.
- Extract and split text from WHO and SATS PDFs.
- Apply length filtering.
- Remove near-duplicate sections with MinHash.
- Emit one shared envelope schema for cases and guideline sections.

## Architecture

```text
2_data_cleaning/
|-- common.py           # regex, schema, JSONL, MinHash, lazy vLLM client
|-- prompts.py          # prompt templates and JSON schemas
|-- clean_mietic.py     # MIETIC parquet to JSONL
|-- clean_pdf.py        # PDF to section JSONL
|-- run_all.py          # stage orchestrator
`-- requirements.txt

data/processed/
|-- mietic.jsonl
|-- sats_sections.jsonl
|-- imci_sections.jsonl
`-- etat_sections.jsonl
```

`vllm` is imported lazily so local Windows runs can test regex and PDF code without CUDA.

## Unified Record Schema

Each output row uses the same envelope. The `case` payload is populated for MIETIC records and the `section` payload is populated for guideline sections.

```jsonc
{
  "id": "<source>_<idx>",
  "source": "mietic | sats | who_imci | who_etat",
  "doc_type": "case | guideline_section",
  "raw_text": "<source text>",
  "case": {},
  "section": {},
  "provenance": {
    "file": "...",
    "row_idx": 0,
    "page_start": 0,
    "page_end": 0,
    "raw_text_sha256": "<16 hex>"
  },
  "license": "<source license>",
  "citation": "<source citation>"
}
```

The `case` payload contains task type, patient demographics, vital signs, chief complaint, symptoms, history, reasoning trace, and prompt version. The `section` payload contains section type, header, breadcrumb, and text.

## MIETIC Parsing

Deterministic regex handles fields that have stable textual patterns:

- task type
- age
- sex
- numeric vital signs such as BP, HR, RR, SpO2, and temperature

LLM extraction handles fields requiring semantic interpretation:

- chief complaint
- symptoms
- history

The MIETIC Hugging Face schema is Alpaca-style (`instruction`, `input`, `output`), not structured clinical fields. Step 2 performs the normalization needed by later pipeline stages.

## LLM Extraction Backend

The validated server stack used Qwen3-32B BF16 through vLLM 0.9.2 with xgrammar guided decoding. The model was loaded once and used for offline in-process batching.

Validated environment:

- Python 3.10.12
- vLLM 0.9.2
- torch 2.7.0+cu126
- transformers 4.52.4
- `VLLM_USE_V1=0`
- `VLLM_ATTENTION_BACKEND=XFORMERS`
- `enforce_eager=True`

The selected model is documented in [the preprocessing model study](0_preprocessing_model_selection_study.md).

## PDF Extraction and Chunking

PDF processing uses a hybrid strategy:

1. Extract text page by page with PyMuPDF. `pdfplumber` is kept as fallback.
2. Split content with clinical-heading regex categories such as signs, diagnosis, treatment, triage, complications, and referral.
3. Split sections longer than 3,000 characters with a 1,500-character sliding window and 200-character overlap.
4. Filter sections outside the 50 to 5,000 character range.
5. Remove near duplicates with MinHash LSH.

PyMuPDF is the primary extractor because WHO IMCI and ETAT PDFs contain CID-encoded fonts that produced corrupted `(cid:NNN)` text in pdfminer-based extraction.

## MinHash Deduplication

Deduplication uses `datasketch.MinHashLSH` with:

- `num_perm = 128`
- Jaccard threshold `0.85`
- intra-source duplicate removal

## Output Counts

| Source | Pages | Heading sections | Output records | Skipped |
| --- | ---: | ---: | ---: | ---: |
| SATS | 21 | 23 | 82 | 1 |
| WHO IMCI | 80 | 33 | 119 | 0 |
| WHO ETAT | 83 | 64 | 122 | 5 |

## Execution

```bash
pip install -r 1_data_fetch/requirements.txt
pip install -r 2_data_cleaning/requirements.txt

python 2_data_cleaning/run_all.py
```

Individual scripts can also be run directly:

```bash
python 2_data_cleaning/clean_mietic.py
python 2_data_cleaning/clean_pdf.py --source sats
python 2_data_cleaning/clean_pdf.py --source who_imci
python 2_data_cleaning/clean_pdf.py --source who_etat
```
