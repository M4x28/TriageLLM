"""DoD verification for step 4 output."""
import json
from collections import Counter
from pathlib import Path

aug_dir = Path(__file__).parent.parent / "data" / "augment"
tr = [json.loads(l) for l in (aug_dir / "train.jsonl").open(encoding="utf-8")]
va = [json.loads(l) for l in (aug_dir / "validation.jsonl").open(encoding="utf-8")]

print(f"DoD #2 train: {len(tr)} (target ~33,575)")
print(f"DoD #3 validation: {len(va)} (target ~1,479)")

val_tags = Counter(r["metadata"].get("oversample_tag") for r in va)
val_pass = all(t == "original" for t in val_tags)
print(f"DoD #4 val oversample_tag: {dict(val_tags)} -> {'PASS' if val_pass else 'FAIL'}")

src = Counter(r["metadata"]["source"] for r in tr)
total = len(tr)
ped = src["who_imci"] + src["who_etat"]
adult = src["sats"]
ident = src["identity"]
mietic = src["mietic"]
print(f"DoD #5 distribution: mietic={mietic/total:.1%} ped={ped/total:.1%} adult={adult/total:.1%} identity={ident/total:.1%}")

bad = 0
for r in tr:
    if "messages" not in r or len(r["messages"]) != 3:
        bad += 1; continue
    roles = [m["role"] for m in r["messages"]]
    if roles != ["system", "user", "assistant"]:
        bad += 1
print(f"DoD #6 schema valid: {total - bad}/{total} -> {'PASS' if bad == 0 else 'FAIL'}")

tag_missing = sum(1 for r in tr if not r["metadata"].get("oversample_tag"))
split_missing = sum(1 for r in tr if not r["metadata"].get("split"))
print(f"DoD #7/8 tag missing: {tag_missing} split missing: {split_missing} -> {'PASS' if tag_missing == 0 and split_missing == 0 else 'FAIL'}")

# Pediatric focus
ped_records = [r for r in tr if r["metadata"]["source"] in ("who_imci", "who_etat")]
ped_caregiver_id = [r for r in tr if r["metadata"]["source"] == "identity" and r["metadata"]["style"] == "caregiver_query"]
ped_focus_total = len(ped_records) + len(ped_caregiver_id)
print(f"\nPediatric/infant focus: {ped_focus_total} records ({ped_focus_total/total:.1%})")
print(f"  - IMCI+ETAT: {len(ped_records)}")
print(f"  - identity caregiver: {len(ped_caregiver_id)}")
