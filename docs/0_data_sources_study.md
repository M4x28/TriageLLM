# Data Source Study: TriageLLM Phase 1

## Scope and Constraints

Phase 1 builds an English supervised fine-tuning corpus for clinical triage decision support with mobile/offline deployment as the target.

Constraints:

- **Language:** English only for Phase 1.
- **Cost:** free sources only; no paid APIs or paywalled data.
- **License:** non-commercial/share-alike licenses are acceptable for Phase 1, but their restrictions propagate to derived datasets and weights.
- **Ethics:** only public, de-identified, or publicly released sources.
- **Access:** sources requiring lengthy credentialing are excluded from the first prototype when usable open alternatives exist.

## Source Tiers

| Tier | Description | Use |
| --- | --- | --- |
| Tier 1 | Labeled triage cases or official triage protocols | backbone training and rule grounding |
| Tier 2 | Target-disease material for sub-Saharan Africa | domain coverage for later expansion |
| Tier 3 | Background medical QA and case reports | evaluation or future augmentation, not primary Phase 1 training |

## Source Inventory

| Source | Tier | Type | Records | License | Cost | Language |
| --- | ---: | --- | ---: | --- | --- | --- |
| [MIETIC](https://huggingface.co/datasets/jackf7499/MIETIC) | 1 | structured ESI cases | 9,629 | CC-BY-NC-SA-4.0 | free | EN |
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) | 1 | ED visits and triage acuity | about 448k visits | PhysioNet DUA | free with credentialing | EN |
| [eICU-CRD](https://physionet.org/content/eicu-crd/) | 1 | ICU stays | about 200k | PhysioNet DUA | free with credentialing | EN |
| [WHO IMCI Chart Booklet](https://www.who.int/publications/i/item/9789241506823) | 1 | pediatric decision tree | 1 PDF | WHO CC-BY-NC-SA-3.0-IGO | free | EN |
| [WHO ETAT Manual](https://www.afro.who.int/publications/emergency-triage-assessment-and-treatment-etat) | 1 | pediatric ABCD triage | 1 PDF | WHO CC-BY-NC-SA-3.0-IGO | free | EN |
| [SATS Training Manual](https://emssa.org.za/sats/) | 1 | 4-color triage and TEWS | 1 PDF | EMSSA terms | free | EN |
| WHO malaria, TB, dengue, cholera, HIV, ANC material | 2 | disease guidelines | multiple PDFs | WHO CC-BY-NC-SA-3.0-IGO | free | EN |
| [MSF Clinical Guidelines](https://medicalguidelines.msf.org/en) | 2 | field manual | about 500 HTML pages | MSF educational terms | free | EN |
| [PLOS NTDs via Europe PMC](https://europepmc.org) | 2 | open-access article metadata | about 200 papers | mostly CC-BY | free | EN |
| [PMC-Patients v2](https://huggingface.co/datasets/zhengyun21/PMC-Patients) | 3 | case report summaries | 250,294 | CC-BY-NC-SA-4.0 | free | EN |
| [MedQA](https://huggingface.co/datasets/bigbio/med_qa) | 3 | medical multiple-choice QA | about 12,700 | MIT | free | EN |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | 3 | medical multiple-choice QA | 193,155 | Apache-2.0 | free | EN |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | 3 | yes/no/maybe biomedical QA | 1k labeled plus unlabeled | MIT | free | EN |
| [HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | 3 | consumer medical dialogue | 112,165 | Apache-2.0 | free | EN |

## Source Assessment

| Source | Strengths | Limitations |
| --- | --- | --- |
| MIETIC | public English ESI-labeled triage cases; manageable size; SFT-friendly Alpaca format | no structured vital-sign fields; requires parsing from narrative input |
| MIMIC-IV-ED | large real-world ED dataset with acuity labels | credentialing delay; US urban domain differs from low-resource deployment target |
| WHO IMCI | canonical pediatric decision tree for common childhood illness | PDF extraction and section splitting required |
| WHO ETAT | pediatric emergency ABCD triage with African-region relevance | semantic overlap with IMCI requires deduplication |
| SATS manual | defines the target SATS label space and TEWS language | spread-layout PDF requires careful extraction |
| MSF guidelines | field-tested LMIC clinical material | broad HTML scrape and maintenance burden |
| PMC-Patients | high-volume case-report summaries | no triage labels; unsuitable as gold triage data |
| MedQA / MedMCQA / PubMedQA | standard external evaluation material | should not be mixed into SFT training |
| HealthCareMagic | large Q&A-style consumer medical dialogue | variable clinical quality and no triage labels |

## Phase 1 Shortlist

Included:

- **MIETIC:** primary labeled ESI case source.
- **SATS manual:** authoritative SATS/TEWS reference.
- **WHO IMCI:** pediatric protocol content.
- **WHO ETAT:** pediatric emergency triage content.

Excluded from Phase 1 training:

- **MIMIC-IV-ED and eICU-CRD:** credentialing delay and domain mismatch.
- **WHO Tier 2, PLOS NTDs, MSF guidelines:** useful for future coverage, not required for the first triage-label prototype.
- **PMC-Patients:** useful for future augmentation, but unlabeled for triage.
- **MedQA, MedMCQA, PubMedQA:** evaluation only.
- **Consumer medical dialogue datasets:** insufficient triage labeling and weaker clinical control.

## Expected Prototype Volume

Estimated pre-clean volume:

| Source | Expected records |
| --- | ---: |
| MIETIC | 9,629 |
| WHO IMCI | about 80 useful clinical sections |
| WHO ETAT | about 60 useful clinical sections |
| SATS manual | about 40 useful sections |

Expected post-clean size: about 9,000 records. After Q&A rewrite and augmentation, the target SFT size is 30,000 to 50,000 examples, suitable for LoRA adaptation of 1.5B to 4B parameter models.
