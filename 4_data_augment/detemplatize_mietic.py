"""Root-data fix for the pediatric ESI-framework leak (round-5).

Diagnosis (Petri found, Bloom confirmed, round-4 additive seeds failed): the model
learned an adult-ED "ESI + resource-prediction" TEMPLATE from the ~28.9k MIETIC
records and applies it to ANY case with vital signs / ED phrasing — including
under-5 low-resource pediatric cases, where the deployed policy requires WHO
IMCI/ETAT (`Action: REFER NOW`, no ESI/SATS, no resource count). Adding clean
counter-examples did not overcome the prior because the leak fires exactly on the
adult-style inputs the MIETIC bulk trained.

This transform attacks the prior at the data root WITHOUT discarding the adult
ESI signal (ESI stays valid for adult ED). It does two things to the MIETIC slice
of the rewrite corpus:

1. DE-TEMPLATIZE (all MIETIC): strip the rote ED resource-prediction block
   ("Predicted Number of Resources", "(Reasoning|Explanation) and Anticipated
   Resources", "Laboratory Tests", "Imaging Studies", "...likely to require N
   resources during the ED visit"). Keep the clinical analysis and the final
   "Triage recommendation: ESI N" line. This removes the distinctive leaked
   sub-template while leaving the adult ESI label intact and aligns MIETIC with
   the system prompt (resources only after the action, if at all).

2. REFRAME UNDER-5 (the few genuine pediatric MIETIC cases): drop the ESI/SATS
   code, set the IMCI/ETAT framework, and — for danger-sign cases (ESI 1-2) —
   lead with `Action: REFER NOW` in IMCI wording with the explicit no-code line.

Adult records are NOT rewritten into IMCI: the frameworks are kept separate.

Idempotent, no LLM, deterministic. Output is a new file so the original rewrite
corpus is preserved.

Usage (server):
  python 4_data_augment/detemplatize_mietic.py \
    --in data/rewrite/train.jsonl --out data/rewrite/train.detemplatized.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

MIETIC_SOURCES = {"mietic"}

# --- resource-prediction template markers -----------------------------------
# Header that opens the resource block (numbered "2." or bold or plain).
_RES_HDR = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?\*{0,2}\s*"
    r"(?:predicted number of resources"
    r"|(?:reasoning|explanation)\s+and\s+anticipated resources"
    r"|anticipated resources"
    r"|resource utilization)"
    r"\b.*$"
)
# Final triage line to preserve, e.g. "**Triage recommendation**: ESI 3 / SATS Yellow".
_TRIAGE_LINE = re.compile(r"(?im)^.*triage recommendation.*$")
# Inline resource-count sentence.
_RES_SENT = re.compile(
    r"(?i)[^.\n]*\b(?:likely to require|will (?:likely )?require|anticipated to use|"
    r"expected to (?:require|use))\b[^.\n]*\bresources?\b[^.\n]*\.")
# Leftover bold resource sub-bullets if any remain after the block cut.
_RES_BULLET = re.compile(
    r"(?im)^\s*[-*]?\s*\*{0,2}\s*(?:laboratory tests?|imaging studies?|imaging|"
    r"lab(?:oratory)? work|blood tests?|ct scan|x-ray|ultrasound)\b.*$")
_BLANKS = re.compile(r"\n{3,}")

# --- under-5 detection -------------------------------------------------------
# STRICT: the age must be the patient's age (require an "old" suffix), so symptom
# DURATIONS like "3 month history" / "4 years ago" do NOT match. Only unambiguous
# pediatric nouns count on their own.
_AGE_YEARS = re.compile(r"\b([0-4])\s*(?:-|\s)?\s*(?:year|yr)s?[-\s]?old\b", re.I)
_AGE_MONTHS = re.compile(r"\b(\d{1,2})\s*(?:-|\s)?\s*(?:month|mo)s?[-\s]?old\b", re.I)
_AGE_WEEKS = re.compile(r"\b\d{1,2}\s*(?:-|\s)?\s*(?:week|day)s?[-\s]?old\b", re.I)
_PEDS_WORD = re.compile(r"\b(?:newborn|neonate|infant|toddler)\b", re.I)
_ESI_SATS_LINE = re.compile(r"(?i)\bESI\s*\d\b.*?(?:/\s*SATS\s*\w+)?")

_REFER = ("Immediate referral to the nearest health facility is required. "
          "Do not wait at home.")
_NO_CODE = ("No ESI or SATS code is assigned because this is an under-5 child in "
            "a low-resource setting (WHO IMCI/ETAT framework).")


def is_under5(text: str) -> bool:
    if _AGE_YEARS.search(text):
        return True
    if _AGE_MONTHS.search(text):
        return int(_AGE_MONTHS.search(text).group(1)) <= 59
    if _AGE_WEEKS.search(text):
        return True
    if _PEDS_WORD.search(text):
        return True
    return False


def detemplatize(ans: str) -> str:
    """Strip the ED resource-prediction block; keep analysis + triage line."""
    triage = _TRIAGE_LINE.search(ans)
    triage_txt = triage.group(0).strip() if triage else None
    hdr = _RES_HDR.search(ans)
    body = ans
    if hdr:
        head = ans[: hdr.start()].rstrip()
        tail = ""
        if triage_txt and triage and triage.start() > hdr.start():
            tail = "\n\n" + triage_txt
        body = head + tail
    body = _RES_SENT.sub("", body)
    body = _RES_BULLET.sub("", body)
    body = _BLANKS.sub("\n\n", body).strip()
    return body


def reframe_under5(ans: str, esi: int | None) -> str:
    """Drop ESI/SATS; IMCI framing. Danger-sign (ESI 1-2) -> REFER NOW lead."""
    body = _ESI_SATS_LINE.sub("", ans)
    body = re.sub(r"(?im)^.*triage recommendation.*$", "", body)
    body = _BLANKS.sub("\n\n", body).strip()
    if esi in (1, 2):
        return f"Action: REFER NOW\n\nThis child has an IMCI/ETAT danger sign. {_REFER}\n\n{body}\n\n{_NO_CODE}"
    return f"{body}\n\n{_NO_CODE}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = Counter()
    out_records = []
    for line in Path(args.inp).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        md = r.get("metadata", {})
        if md.get("source") not in MIETIC_SOURCES:
            out_records.append(r)
            stats["non_mietic_kept"] += 1
            continue
        msgs = r.get("messages") or []
        if not msgs:
            out_records.append(r)
            continue
        ans = msgs[-1].get("content", "")
        q = " ".join(m.get("content", "") for m in msgs[:-1])
        had_block = bool(_RES_HDR.search(ans) or _RES_SENT.search(ans))

        if is_under5(q + " " + ans):
            esi = md.get("esi")
            esi = int(esi) if isinstance(esi, (int, str)) and str(esi).isdigit() else None
            new = reframe_under5(detemplatize(ans), esi)
            md.update(triage_framework="IMCI_ETAT")
            md.pop("esi", None)
            if esi in (1, 2):
                md["action"] = "REFER NOW"
            md["detemplatized"] = "under5_reframe"
            stats["under5_reframed"] += 1
        else:
            new = detemplatize(ans)
            md["detemplatized"] = "adult_strip" if had_block else "adult_nochange"
            stats["adult_detemplatized" if had_block else "adult_unchanged"] += 1

        msgs[-1]["content"] = new
        r["metadata"] = md
        out_records.append(r)

    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
        encoding="utf-8")
    print(f"wrote {len(out_records)} records -> {args.out}")
    for k, v in stats.most_common():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
