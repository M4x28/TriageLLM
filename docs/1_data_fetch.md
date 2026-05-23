# Step 1: Data Fetch

## Scope

Step 1 downloads the raw bytes for the Phase 1 source shortlist. It does not parse PDF text, normalize schemas, extract clinical fields, or deduplicate content. Those operations belong to Step 2.

| Source key | Source type          | Output                          | License family          |
| ---------- | -------------------- | ------------------------------- | ----------------------- |
| `mietic`   | Hugging Face dataset | `data/raw/mietic/train.parquet` | CC-BY-NC-SA-4.0         |
| `sats`     | HTTP PDF             | `data/raw/pdf/sats_manual.pdf`  | EMSSA terms             |
| `who_imci` | HTTP PDF             | `data/raw/pdf/who_imci.pdf`     | WHO CC-BY-NC-SA-3.0-IGO |
| `who_etat` | HTTP PDF             | `data/raw/pdf/who_etat.pdf`     | WHO CC-BY-NC-SA-3.0-IGO |

## Architecture

```text
1_data_fetch/
|-- common.py          # shared helpers for download, hashing, and manifest writes
|-- fetch_mietic.py    # MIETIC dataset fetcher
|-- fetch_sats.py      # SATS PDF fetcher
|-- fetch_imci.py      # WHO IMCI PDF fetcher
|-- fetch_etat.py      # WHO ETAT PDF fetcher
`-- requirements.txt

data/raw/
|-- manifest.json
|-- mietic/
`-- pdf/
```

The entrypoint scripts are thin source-specific wrappers. Download, retry, SHA-256 hashing, and manifest update logic are centralized in `1_data_fetch/common.py`.

## Manifest

`data/raw/manifest.json` records one entry for each downloaded artifact. The manifest is used for reproducibility, auditability, and cache validation.

```json
{
  "source": "mietic",
  "path": "mietic/train.parquet",
  "url": "hf://jackf7499/MIETIC",
  "sha256": "<64 hex>",
  "size_bytes": 7560291,
  "fetched_at": "2026-05-19T00:07:40Z",
  "license": "CC-BY-NC-SA-4.0"
}
```

Manifest writes are atomic: a temporary file is written on the same volume and then moved into place with `os.replace`.

## Idempotency

Each fetcher checks:

1. The target file exists.
2. The manifest contains an entry for the file.
3. The current SHA-256 matches the manifest SHA-256.

If all three checks pass, the script exits without downloading. Missing files, missing manifest entries, or hash mismatches trigger a new download.

## Network Handling

HTTP downloads stream to disk and retry three times with exponential backoff. Minimum-size checks reject common failure modes such as HTML error pages returned with HTTP 200.

## Execution

```powershell
cd TriageLLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r 1_data_fetch\requirements.txt

python 1_data_fetch\fetch_mietic.py
python 1_data_fetch\fetch_sats.py
python 1_data_fetch\fetch_imci.py
python 1_data_fetch\fetch_etat.py
```

## Initial Run Artifacts

| Source     |   Size | SHA-256 prefix |  Rows |
| ---------- | -----: | -------------- | ----: |
| `mietic`   | 7.2 MB | `ae7e698c55aa` | 9,629 |
| `sats`     | 1.6 MB | `410c8d4b11b5` |   n/a |
| `who_imci` | 5.1 MB | `d10fd1d040bd` |   n/a |
| `who_etat` | 597 KB | `9f2c85bf9592` |   n/a |
