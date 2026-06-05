"""Verify target-prompt integrity in Petri `.eval` logs.

Petri's auditor stages a target system message during scenario setup (visible in
the auditor view as `set_system_message`). Our custom target agent
(`triage_target.py`) consumes and DISCARDS that staged message and forces the
deployed `SYSTEM_PROMPT` (v2) instead. This script proves, from the logs, that
the override held: every system message the TARGET model actually received must
equal `SYSTEM_PROMPT` exactly, and there must be exactly ONE distinct target
system prompt across the whole run.

Run this after every Petri run (pilot AND full) as a gate: if it FAILs, the
behavioral findings are NOT attributable to the model (possible prompt-injection
/ setup leak) and must be discarded.

Usage:
  python 6_evaluation/petri/verify_target_prompt.py --model qwen3-1.7b
Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RESULTS_ROOT = REPO_ROOT / "data" / "eval" / "petri"

sys.path.insert(0, str(HERE.parent / "static_evaluation"))
from prompts import SYSTEM_PROMPT  # noqa: E402

EXPECTED = SYSTEM_PROMPT.strip()
EXPECTED_HASH = hashlib.md5(EXPECTED.encode()).hexdigest()[:12]


def _text(content) -> str:
    return content if isinstance(content, str) else str(content)


def _resolve(content: str, attachments: dict) -> str:
    """Resolve an inspect attachment reference to its stored content."""
    if isinstance(content, str) and content.startswith("attachment://"):
        return attachments.get(content.split("//", 1)[1], content)
    return content


def _is_target_event(ev) -> bool:
    if getattr(ev, "event", "") != "model":
        return False
    model = getattr(ev, "model", "") or ""
    return "target" in model or "1.7b" in model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from inspect_ai.log import read_eval_log

    model_dir = RESULTS_ROOT / args.model
    files = sorted(glob.glob(str(model_dir / "*.eval")))
    if not files:
        print(f"FAIL: no .eval logs under {model_dir}")
        return 1

    distinct: set[str] = set()
    n_msgs = 0
    bad: list[str] = []

    for f in files:
        log = read_eval_log(f)
        for s in (log.samples or []):
            att = s.attachments or {}
            for ev in (s.events or []):
                if not _is_target_event(ev):
                    continue
                for m in (getattr(ev, "input", None) or []):
                    if getattr(m, "role", "") != "system":
                        continue
                    raw = _resolve(_text(getattr(m, "content", "")), att)
                    txt = _text(raw).strip()
                    n_msgs += 1
                    distinct.add(hashlib.md5(txt.encode()).hexdigest()[:12])
                    if txt != EXPECTED:
                        bad.append(f"{s.id} epoch={s.epoch}: len={len(txt)} "
                                   f"head={txt[:60]!r}")

    print(f"target system messages checked : {n_msgs}")
    print(f"distinct target system prompts : {len(distinct)} {sorted(distinct)}")
    print(f"expected v2 hash               : {EXPECTED_HASH}")
    if n_msgs == 0:
        print("FAIL: no target system messages found in logs")
        return 1
    if bad:
        print(f"FAIL: {len(bad)} target system message(s) != SYSTEM_PROMPT v2")
        for b in bad[:10]:
            print("  -", b)
        return 1
    if distinct != {EXPECTED_HASH}:
        print("FAIL: target saw a system prompt that is not exactly v2")
        return 1
    print("PASS: every target system message == SYSTEM_PROMPT v2 "
          "(single distinct prompt). Findings are attributable to the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
