"""Orchestrator: clean_mietic + 3× clean_pdf in sequence.

Designed to run on the A100 server where vLLM is available.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import setup_logging


SCRIPTS = [
    ("mietic", ["clean_mietic.py"]),
    ("sats",   ["clean_pdf.py", "--source", "sats"]),
    ("imci",   ["clean_pdf.py", "--source", "imci"]),
    ("etat",   ["clean_pdf.py", "--source", "etat"]),
]


def main() -> int:
    log = setup_logging("triagellm.clean.run_all")
    here = Path(__file__).resolve().parent
    failures = []

    for name, argv in SCRIPTS:
        log.info("=" * 60)
        log.info("running %s", " ".join(argv))
        log.info("=" * 60)
        cmd = [sys.executable, str(here / argv[0]), *argv[1:]]
        rc = subprocess.call(cmd, cwd=here)
        if rc != 0:
            log.error("%s failed (rc=%d)", name, rc)
            failures.append(name)

    log.info("=" * 60)
    log.info("done. failures: %s", failures or "none")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
