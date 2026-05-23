"""Print 1 sample record per style from train_cases.jsonl."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "rewrite" / "train_cases.jsonl"
records = [json.loads(l) for l in DATA.open()]
print(f"total: {len(records)}")
seen = set()
for r in records:
    s = r["metadata"]["style"]
    if s in seen:
        continue
    seen.add(s)
    print(f"\n=== {s} ===")
    print(f"id: {r['id']}")
    print(f"task_type: {r['metadata']['task_type']}")
    print(f"QUESTION: {r['messages'][1]['content']}")
    print(f"ANSWER (first 250 chars):")
    print(r['messages'][2]['content'][:250])
