"""Orchestrator: rewrite_cases + rewrite_sections + merge.

Reuses the same vLLM env (set by orchestrate.sh wrapper).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from common import REWRITE_DIR, setup_logging


SCRIPTS = [
    ("cases",    ["rewrite_cases.py"]),
    ("sections", ["rewrite_sections.py", "--source", "all"]),
]


def merge_outputs(log) -> int:
    out = REWRITE_DIR / "train.jsonl"
    sources = [
        REWRITE_DIR / "train_cases.jsonl",
        REWRITE_DIR / "train_sections.jsonl",
    ]
    n = 0
    with out.open("w", encoding="utf-8") as fp:
        for src in sources:
            if not src.exists():
                log.warning("missing %s — skip", src)
                continue
            with src.open(encoding="utf-8") as f:
                for line in f:
                    fp.write(line)
                    n += 1
    log.info("merged → %s (%d records total)", out, n)
    return n


def main() -> int:
    log = setup_logging("triagellm.rewrite.run_all")
    here = Path(__file__).resolve().parent
    failures = []

    REWRITE_DIR.mkdir(parents=True, exist_ok=True)

    for name, argv in SCRIPTS:
        log.info("=" * 60)
        log.info("running %s", " ".join(argv))
        log.info("=" * 60)
        cmd = [sys.executable, str(here / argv[0]), *argv[1:]]
        rc = subprocess.call(cmd, cwd=here)
        if rc != 0:
            log.error("%s failed (rc=%d)", name, rc)
            failures.append(name)

    if not failures:
        merge_outputs(log)

    log.info("=" * 60)
    log.info("done. failures: %s", failures or "none")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
