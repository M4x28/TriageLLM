"""Publish the quantized GGUF to a PRIVATE HuggingFace Hub repo.

Single owner, no collaborators, no search indexing. The repo is meant
to be loaded by PocketPal (or any llama.cpp client) using the owner's
HF token.

Usage:
  python 7_deploy/publish_to_hf.py \
      --gguf data/deploy/gguf/qwen3-1.7b-q4_k_m.gguf \
      --repo <username>/triagellm-qwen3-1.7b-gguf \
      --private
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

from common import COMPARE_FILE, EVAL_Q4KM_DIR, INTERNAL_DIR, setup_logging


MODEL_CARD_TEMPLATE = """\
---
license: cc-by-nc-sa-4.0
language:
  - en
tags:
  - triage
  - medical
  - clinical-decision-support
  - llama-cpp
  - gguf
  - q4_k_m
  - quantized
  - lmic
base_model: Qwen/Qwen3-1.7B
library_name: gguf
inference: false
---

# TriageLLM Qwen3-1.7B (Q4_K_M GGUF)

Educational / humanitarian fine-tune of Qwen3-1.7B for clinical triage
decision support in low-resource settings (sub-Saharan Africa primary
target). Quantized to Q4_K_M (~1 GB) for smartphone deployment via
llama.cpp.

## NOT a medical device

This model is **decision support only**. It is **not** a substitute for a
qualified clinician, not certified for clinical use, and must always be
used behind a UI disclaimer explaining this. The author and the upstream
data providers assume no liability for clinical decisions based on
model outputs.

## Intended use

* Educational reference on WHO IMCI / ETAT pediatric triage protocols.
* SATS (South Africa Triage Scale) colour assignment.
* ESI (Emergency Severity Index) 1-5 classification.
* Pilot deployment with NGO partners under clinician supervision.

## Out-of-scope

* Autonomous diagnosis without clinician review.
* Drug dosing in unsupervised contexts.
* Pediatric < 3 months (TEWS infant path not validated).
* Commercial use (license CC-BY-NC-SA-4.0 propagated from MIETIC).

## Training summary

* Base: Qwen/Qwen3-1.7B (Apache-2.0).
* SFT: LoRA r=16 alpha=32, 3 epochs, lr 2e-4, bf16, 33,575 records.
* Dataset sources: MIETIC (CC-BY-NC-SA-4.0), WHO IMCI + ETAT, SATS,
  MSF Clinical Guidelines, WHO Tier-2 disease guidelines, PLOS NTDs.

## Quantization

* Format: GGUF Q4_K_M (4.5 bits per weight, per-group with double-quant
  scales). Generated with `llama.cpp` `llama-quantize`.
* Size: ~1.0 GB.
* Target hardware: Android smartphone, 4 GB RAM (Snapdragon 7-gen or
  equivalent), ARM Neon SIMD.

## Recommended system prompt

```
{system_prompt}
```

## Evaluation summary

Validation set: 1,479 records (step 4, no oversampling).

### BF16 baseline (step 6)

* Validation NLL: {bf16_nll} (PPL {bf16_ppl})
* Triage ESI accuracy: {bf16_esi}
* Triage SATS accuracy: {bf16_sats}
* Pediatric recall (gold keyword overlap): {bf16_pediatric}
* Refusal rate (10 identity probes): {bf16_refusal}

### Q4_K_M post-quantization (step 7)

* Validation NLL: {q4km_nll} (PPL {q4km_ppl})  delta {nll_delta}
* Triage ESI accuracy: {q4km_esi}  delta {esi_delta}
* Triage SATS accuracy: {q4km_sats}  delta {sats_delta}
* Pediatric recall: {q4km_pediatric}  delta {pediatric_delta}
* Refusal rate: {q4km_refusal}  delta {refusal_delta}

Gate decision: **{decision}**.

## Inference

### llama.cpp CLI

```bash
./llama-cli -m qwen3-1.7b-q4_k_m.gguf \\
    --system "{system_prompt_short}" \\
    -p "Patient: 4y old, RR 60, chest indrawing. Triage colour?" -n 256
```

### PocketPal (Android)

1. Install PocketPal AI from the Play Store.
2. Settings -> Add model -> HuggingFace -> paste this repo URL.
3. Authenticate with your HF token (private repo).
4. Paste the recommended system prompt above.
5. Chat.

## License

CC-BY-NC-SA-4.0. Inherited from the MIETIC training dataset license.
Non-commercial use only. Derivatives must share alike.

## Citation

If you use this model, please reference both this card and the MIETIC
dataset (Jack Foster et al., HF `jackf7499/MIETIC`).
"""


def _checksum_sha256(path: Path) -> str:
    """Return sha256 of the file content (streamed)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_model_card(slug: str, system_prompt: str,
                      bf16: dict, q4km: dict,
                      cmp: dict | None) -> str:
    """Render the README model card with the latest eval numbers."""
    tri_bf = bf16["triage_and_latency"]
    tri_q4 = q4km["triage_and_latency"]
    bf_id = bf16["identity"]
    q4_id = q4km["identity"]

    nll_delta = (q4km["validation"]["nll"] - bf16["validation"]["nll"])
    esi_delta = (tri_q4["esi"]["accuracy"] - tri_bf["esi"]["accuracy"])
    sats_delta = (tri_q4["sats"]["accuracy"] - tri_bf["sats"]["accuracy"])
    pediatric_delta = (tri_q4["pediatric"]["recall"]
                       - tri_bf["pediatric"]["recall"])
    refusal_delta = q4_id["refusal_rate"] - bf_id["refusal_rate"]
    decision = (cmp or {}).get("decision", "unknown")

    return MODEL_CARD_TEMPLATE.format(
        system_prompt=system_prompt.strip(),
        system_prompt_short=system_prompt.strip().replace('"', '\\"')[:200],
        bf16_nll=bf16["validation"]["nll"],
        bf16_ppl=bf16["validation"]["perplexity"],
        bf16_esi=tri_bf["esi"]["accuracy"],
        bf16_sats=tri_bf["sats"]["accuracy"],
        bf16_pediatric=tri_bf["pediatric"]["recall"],
        bf16_refusal=bf_id["refusal_rate"],
        q4km_nll=q4km["validation"]["nll"],
        q4km_ppl=q4km["validation"]["perplexity"],
        q4km_esi=tri_q4["esi"]["accuracy"],
        q4km_sats=tri_q4["sats"]["accuracy"],
        q4km_pediatric=tri_q4["pediatric"]["recall"],
        q4km_refusal=q4_id["refusal_rate"],
        nll_delta=round(nll_delta, 4),
        esi_delta=round(esi_delta, 4),
        sats_delta=round(sats_delta, 4),
        pediatric_delta=round(pediatric_delta, 4),
        refusal_delta=round(refusal_delta, 4),
        decision=decision,
    )


def main() -> int:
    from common import SYSTEM_PROMPT  # imported here to keep CLI fast

    log = setup_logging("triagellm.deploy.publish")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True,
                    help="path to the GGUF file to upload")
    ap.add_argument("--repo", required=True,
                    help="HF repo id, e.g. '<username>/triagellm-qwen3-1.7b-gguf'")
    ap.add_argument("--model-slug", default="qwen3-1.7b")
    ap.add_argument("--private", action="store_true", default=True,
                    help="create the repo as PRIVATE (default). Use --public to override.")
    ap.add_argument("--public", action="store_true",
                    help="opt out of private (NOT recommended, license issues)")
    ap.add_argument("--token", default=None,
                    help="HF token (default: read from HF_TOKEN env or HF cache)")
    args = ap.parse_args()

    is_private = not args.public

    from huggingface_hub import HfApi, create_repo, upload_file

    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        log.error("GGUF file not found: %s", gguf_path)
        return 1

    log.info("checksum sha256 of %s (this may take a few seconds)", gguf_path)
    sha = _checksum_sha256(gguf_path)
    log.info("sha256 = %s", sha)

    log.info("creating private=%s repo %s", is_private, args.repo)
    create_repo(args.repo, private=is_private, repo_type="model",
                exist_ok=True, token=args.token)

    bf16_path = INTERNAL_DIR / f"{args.model_slug}.json"
    q4km_path = EVAL_Q4KM_DIR / f"{args.model_slug}.json"
    bf16 = json.loads(bf16_path.read_text(encoding="utf-8"))
    q4km = json.loads(q4km_path.read_text(encoding="utf-8"))
    cmp = None
    if COMPARE_FILE.exists():
        cmp = json.loads(COMPARE_FILE.read_text(encoding="utf-8"))

    log.info("rendering model card")
    card = _build_model_card(args.model_slug, SYSTEM_PROMPT, bf16, q4km, cmp)

    # Upload artifacts: GGUF + README + sha256 + eval JSONs for transparency.
    artifacts = [
        (gguf_path, gguf_path.name),
        (bf16_path, f"eval/bf16-{args.model_slug}.json"),
        (q4km_path, f"eval/q4km-{args.model_slug}.json"),
    ]
    if COMPARE_FILE.exists():
        artifacts.append((COMPARE_FILE, "eval/compare_bf16_q4km.json"))

    for local, remote in artifacts:
        log.info("uploading %s -> %s", local, remote)
        upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                    repo_id=args.repo, repo_type="model", token=args.token)

    # README.md and checksums.txt are uploaded inline.
    api = HfApi()
    log.info("uploading README.md (model card)")
    api.upload_file(path_or_fileobj=card.encode("utf-8"),
                    path_in_repo="README.md", repo_id=args.repo,
                    repo_type="model", token=args.token)

    log.info("uploading checksums.txt")
    checksum_content = f"{sha}  {gguf_path.name}\n".encode("utf-8")
    api.upload_file(path_or_fileobj=checksum_content,
                    path_in_repo="checksums.txt", repo_id=args.repo,
                    repo_type="model", token=args.token)

    log.info("publish complete. Repo: https://huggingface.co/%s "
             "(visibility=%s)", args.repo,
             "private" if is_private else "public")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
