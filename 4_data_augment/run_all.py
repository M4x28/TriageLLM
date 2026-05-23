"""Step 4 orchestrator — runs augment.py + light sanity check on output."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from common import AUGMENT_DIR, setup_logging


def _sanity_check(path: Path) -> dict:
    seen_keys = set()
    seen_split = set()
    seen_tag = set()
    by_source: dict[str, int] = {}
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            seen_keys.update(r.keys())
            m = r["metadata"]
            seen_split.add(m.get("split"))
            seen_tag.add(m.get("oversample_tag"))
            by_source[m.get("source", "?")] = by_source.get(m.get("source", "?"), 0) + 1
            if r["messages"][0]["role"] != "system":
                raise AssertionError(f"record {r.get('id')}: first role not system")
            if r["messages"][-1]["role"] != "assistant":
                raise AssertionError(f"record {r.get('id')}: last role not assistant")
    return {
        "lines": n,
        "top_keys": sorted(seen_keys),
        "splits": sorted(seen_split, key=lambda x: x or ""),
        "oversample_tags_seen": sorted(seen_tag, key=lambda x: x or ""),
        "by_source": by_source,
    }


def main() -> int:
    log = setup_logging("triagellm.augment.run_all")

    step_dir = Path(__file__).resolve().parent
    augment = step_dir / "augment.py"

    cmd = [sys.executable, str(augment)]
    if "--skip-identity" in sys.argv:
        cmd.append("--skip-identity")
    log.info("running %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        log.error("augment.py exited %d", proc.returncode)
        return proc.returncode

    train = AUGMENT_DIR / "train.jsonl"
    val = AUGMENT_DIR / "validation.jsonl"
    log.info("sanity-check %s", train)
    train_info = _sanity_check(train)
    log.info("train: %s", json.dumps(train_info, indent=2))
    log.info("sanity-check %s", val)
    val_info = _sanity_check(val)
    log.info("validation: %s", json.dumps(val_info, indent=2))

    val_copies = [t for t in val_info["oversample_tags_seen"] if t and t != "original"]
    if val_copies:
        log.error("validation set contains oversample copies: %s", val_copies)
        return 1
    log.info("validation set is original-only — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
