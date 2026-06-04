"""Run one Bloom behavioral evaluation against a local target model.

Thin orchestration only - it does not reimplement Bloom. It (1) configures the
local endpoints, (2) injects our triage SYSTEM_PROMPT into the target as
`target_sysprompt_prefix` (single source: read from static_evaluation/prompts.py,
written into a gitignored work copy of the behavior so the seed stays pristine),
(3) runs `bloom scenarios`, (4) runs `inspect eval petri_bloom/bloom_audit`,
writing the NATIVE inspect `.eval` log under
`data/eval/behavioral/<model>/<behavior>/`.

Prereqs: target and auditor/judge served via `serve_model.sh`. Run in `.venv-bloom`.

Usage:
  python 6_evaluation/behavioral/run_behavioral.py --model qwen3-1.7b --behavior false_reassurance
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import endpoints

HERE = Path(__file__).resolve().parent
SEEDS_DIR = HERE / "seeds"
WORK_DIR = HERE / ".work"          # gitignored; injected prefix + scenarios live here
REPO_ROOT = HERE.parent.parent
RESULTS_ROOT = REPO_ROOT / "data" / "eval" / "behavioral"

# Single source of truth for the target's system prompt.
sys.path.insert(0, str(HERE.parent / "static_evaluation"))
from prompts import SYSTEM_PROMPT  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("[run_behavioral] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _prepare_work_behavior(behavior: str) -> Path:
    """Copy the seed BEHAVIOR.md into a work dir with SYSTEM_PROMPT injected as
    `target_sysprompt_prefix`. Keeps the committed seed prompt-free."""
    seed_md = (SEEDS_DIR / behavior / "BEHAVIOR.md").read_text(encoding="utf-8")
    # frontmatter is between the first two '---' fences
    _, frontmatter, body = seed_md.split("---", 2)
    indented = "\n".join("  " + ln for ln in SYSTEM_PROMPT.split("\n"))
    new_fm = f"{frontmatter.rstrip()}\ntarget_sysprompt_prefix: |\n{indented}\n"
    work = WORK_DIR / behavior
    work.mkdir(parents=True, exist_ok=True)
    (work / "BEHAVIOR.md").write_text(f"---{new_fm}---{body}", encoding="utf-8")
    return work


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="target served-model-name (e.g. qwen3-1.7b)")
    ap.add_argument("--behavior", required=True,
                    help="behavior name; must match a dir under seeds/")
    ap.add_argument("--auditor-model", default=endpoints.DEFAULT_AUDITOR.model)
    ap.add_argument("--target-port", type=int, default=endpoints.DEFAULT_TARGET.port)
    ap.add_argument("--auditor-port", type=int, default=endpoints.DEFAULT_AUDITOR.port)
    ap.add_argument("--regen-scenarios", action="store_true")
    args = ap.parse_args()

    if not (SEEDS_DIR / args.behavior / "BEHAVIOR.md").exists():
        print(f"ERROR: no seed BEHAVIOR.md for '{args.behavior}'", file=sys.stderr)
        return 1

    # Work copy with the triage SYSTEM_PROMPT injected as the target prefix.
    work = _prepare_work_behavior(args.behavior)

    target = endpoints.Endpoint("target", args.model, args.target_port)
    auditor = endpoints.Endpoint("auditor", args.auditor_model, args.auditor_port)
    judge = endpoints.Endpoint("judge", args.auditor_model, args.auditor_port)
    roles = endpoints.configure_local(target=target, auditor=auditor, judge=judge)

    # 1. Scenario suite (reused unless --regen-scenarios).
    scenarios_dir = work / "scenarios"
    if args.regen_scenarios and scenarios_dir.exists():
        shutil.rmtree(scenarios_dir)
    if not scenarios_dir.exists():
        _run(["bloom", "scenarios", str(work),
              "--model-role", f"scenarios={roles['scenarios']}"])
    else:
        print(f"[run_behavioral] reusing scenarios at {scenarios_dir}")

    # 2. Audit (Rollout + Judgment). Native .eval log -> per-model/behavior dir.
    log_dir = RESULTS_ROOT / args.model / args.behavior
    log_dir.mkdir(parents=True, exist_ok=True)
    _run([
        "inspect", "eval", "petri_bloom/bloom_audit",
        "-T", f"behavior={work}",
        "--model-role", f"auditor={roles['auditor']}",
        "--model-role", f"target={roles['target']}",
        "--model-role", f"judge={roles['judge']}",
        "--log-dir", str(log_dir),
    ])

    print(f"[run_behavioral] done. Native logs in {log_dir}")
    print(f"[run_behavioral] view: inspect view --log-dir {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
