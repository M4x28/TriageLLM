"""Quick inspector for identity_seed.jsonl quality check."""
import json
from collections import defaultdict
from pathlib import Path

p = Path(__file__).parent / "identity_seed.jsonl"
recs = [json.loads(l) for l in p.open(encoding="utf-8")]
print(f"total: {len(recs)}")

g = defaultdict(list)
for r in recs:
    k = (r["metadata"]["style"], r["metadata"]["pattern"])
    g[k].append(r)

for k, group in g.items():
    print(f"\n=== {k} ({len(group)} variants) ===".upper())
    for i, r in enumerate(group[:3]):
        print(f"  Q[{i}]: {r['messages'][1]['content']}")
    print(f"  A: {group[0]['messages'][2]['content'][:220]}...")
