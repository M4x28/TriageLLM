# TriageLLM — Project Context

## Goal

Fine-tune a small (1.5-4B params) LLM to assist medical triage in low-resource
settings (sub-Saharan Africa primary target). Deploy on smartphone via
llama.cpp quantized GGUF Q4_K_M. Educational / humanitarian use (NOT a
certified medical device).

**Disclaimer guarantee:** every model output ships behind a UI disclaimer
that it is decision-support only and never a substitute for a clinician.

## Phases

| Phase | Status | Output |
|-------|--------|--------|
| 1. Data collection + preprocessing | In progress | `data/processed/corpus.jsonl` |
| 2. SFT with LoRA on base model (see `docs/model_selection_study.md`) | Not started | LoRA adapter + merged checkpoint |
| 3. Evaluation (triage accuracy, F1 per class, recall on critical cases) | Not started | Eval report |
| 4. GGUF quantization + mobile deploy via llama.cpp | Not started | Android APK / iOS build |
| 5. Clinical validation pilot with NGO partner | Not started | Field report |

## Stack

- **Training GPU**: NVIDIA L40S 48GB VRAM (Ada Lovelace, bf16 native)
- **Driver**: NVIDIA R570 production (CUDA 12.8) — request pending
- **Python**: 3.11
- **Inference (training)**: vLLM 0.6.6
- **Training**: HF transformers 4.47.1 + accelerate 1.2.1 + peft 0.14.0 + trl 0.13.0
- **Quantization**: llama.cpp (GGUF Q4_K_M for ~1-2 GB mobile footprint)
- **Mobile runtime**: llama.cpp Android (CPU inference)

## Repository layout

```
TriageLLM/
├── AGENTS.md                       # this file
├── .Codex/                        # Codex project config
├── data/                           # Phase 1: data collection
│   ├── requirements.txt
│   ├── download_*.py               # source-specific downloaders
│   ├── extract_pdf_sections.py
│   ├── parse_plos_ntds.py
│   ├── unify_to_jsonl.py           # merges all sources to common schema
│   ├── validate_corpus.py
│   ├── raw/                        # immutable downloaded data
│   │   ├── hf/                     # HuggingFace datasets (parquet)
│   │   ├── who/                    # WHO guideline PDFs
│   │   ├── sats/                   # SATS manual PDF
│   │   ├── msf/                    # MSF scraped HTML
│   │   └── plos_ntds/              # PLOS NTDs paper metadata
│   └── processed/                  # derived (regenerable)
│       ├── who_msf_sections.jsonl
│       └── corpus.jsonl            # unified training corpus
└── docs/
    └── data_collection.md          # Phase 1 doc
```

## Data sources (Phase 1)

| Source | Tier | Records | License |
|--------|------|---------|---------|
| MIETIC (HF `jackf7499/MIETIC`) | 1 | 9,629 cases, ESI 1-5 labeled | CC-BY-NC-SA-4.0 |
| WHO IMCI Chart Booklet | 1 | Pediatric decision tree | WHO CC-BY-NC-SA-3.0-IGO |
| WHO ETAT Manual | 1 | Pediatric emergency triage | WHO CC-BY-NC-SA-3.0-IGO |
| SATS Training Manual 2012 | 1 | 4-color triage + TEWS | EMSSA permission |
| MSF Clinical Guidelines | 1 | Field manual LMIC | MSF educational license |
| WHO Tier-2 disease guidelines | 2 | Malaria, TB, dengue, cholera, yellow fever, HIV, maternal | WHO CC-BY-NC-SA-3.0-IGO |
| PLOS NTDs supplementary | 2 | NTDs Africa (leishmaniasis, schistosomiasis, etc.) | CC-BY-4.0 |
| PMC-Patients | 3 | 167k case reports | CC-BY-NC-SA-4.0 |
| MedQA / MedMCQA | 3 | Eval baseline | MIT |

See `docs/data_collection.md` for full pipeline detail.

## Key clinical decisions

- **Target label space**: SATS 4-color (Red / Orange / Yellow / Green) + TEWS score
- **ESI → SATS mapping**: ESI 1→Red, 2→Orange, 3→Yellow, 4-5→Green
- **TEWS reconciliation**: cross-check ESI mapping with TEWS computed from
  vital signs; if ESI says green but TEWS ≥ 7, flag inconsistency and prefer
  the more severe label
- **Age stratification**: TEWS adult (≥12y) vs paediatric (3mo-12y) vs infant
  (<3mo) thresholds differ — see SATS manual pp. 14-19

## Working conventions for Codex

- **Language for prose/responses**: italiano (per user AGENTS.md global)
- **Code/commits/PRs**: English, normal prose (not caveman)
- **Caveman mode**: active globally (level: full). Strip articles/filler in
  user-facing prose. Code blocks unchanged.
- **Learning output style**: when generating 20+ lines involving clinical
  judgement, request `TODO(human)` contribution from user via Learn by Doing
  block (see `unify_to_jsonl.py::compute_tews` for live example).
- **Plan mode**: write plans to user's plans directory; ExitPlanMode for
  approval before executing.
- **Memory**: persistent .md memory at `C:\Users\leobi\.Codex\projects\...\memory\` —
  consult `MEMORY.md` index at session start.

## Safety / compliance reminders

- **Never claim diagnosis** — model output framed as decision support
- **Always show top-3 with confidence** and "consult clinician" message
- **CC-BY-NC-SA-4.0 license propagates** from MIETIC + PMC-Patients →
  the final model derivative is non-commercial. If commercial use needed,
  rebuild without these sources.
- **PhysioNet DUA**: the HF mirror bypasses our need for credentials,
  but if we ever fetch MIMIC-IV directly, CITI training is required.
- **PII**: MIETIC is already de-identified by source. Do not attempt
  re-identification. Do not redistribute raw records — train and publish
  weights only.

## Out-of-scope guard-rails

- No production deployment without clinical partner accountability
- No claims of "AI doctor" or diagnostic certification
- No drug-dosing without explicit guideline citation in output
- No paediatric < 3 months recommendations unless TEWS infant path implemented
