"""Run one Bloom behavioral evaluation against a local target model.

Thin orchestration only — it does not reimplement Bloom. It (1) configures the
local endpoints, (2) runs `bloom scenarios` to generate the suite (reused on
re-runs unless --regen-scenarios), and (3) runs `inspect eval
petri_bloom/bloom_audit`, writing the NATIVE inspect `.eval` log under
`data/eval/behavioral/<model>/<behavior>/`.

Prereqs: the target and auditor/judge models are already served via
`serve_model.sh` (see README). Run inside `.venv-bloom`.

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
REPO_ROOT = HERE.parent.parent
RESULTS_ROOT = REPO_ROOT / "data" / "eval" / "behavioral"


def _run(cmd: list[str]) -> None:
    print("[run_behavioral] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="target served-model-name (e.g. qwen3-1.7b)")
    ap.add_argument("--behavior", required=True,
                    help="behavior name; must match a dir under seeds/")
    ap.add_argument("--auditor-model", default=endpoints.DEFAULT_AUDITOR.model,
                    help="served-model-name of the auditor/judge model")
    ap.add_argument("--target-port", type=int, default=endpoints.DEFAULT_TARGET.port)
    ap.add_argument("--auditor-port", type=int, default=endpoints.DEFAULT_AUDITOR.port)
    ap.add_argument("--regen-scenarios", action="store_true",
                    help="regenerate the scenario suite even if it already exists")
    args = ap.parse_args()

    behavior_dir = SEEDS_DIR / args.behavior
    if not (behavior_dir / "BEHAVIOR.md").exists():
        print(f"ERROR: no BEHAVIOR.md at {behavior_dir}", file=sys.stderr)
        return 1

    # Endpoints: target = model under test; auditor+judge = strong local model.
    target = endpoints.Endpoint("target", args.model, args.target_port)
    auditor = endpoints.Endpoint("auditor", args.auditor_model, args.auditor_port)
    judge = endpoints.Endpoint("judge", args.auditor_model, args.auditor_port)
    roles = endpoints.configure_local(target=target, auditor=auditor, judge=judge)

    # 1. Scenario suite (Understanding + Ideation). Reused unless --regen.
    scenarios_dir = behavior_dir / "scenarios"
    if args.regen_scenarios and scenarios_dir.exists():
        shutil.rmtree(scenarios_dir)
    if not scenarios_dir.exists():
        _run(["bloom", "scenarios", str(behavior_dir),
              "--model-role", f"scenarios={roles['scenarios']}"])
    else:
        print(f"[run_behavioral] reusing existing scenarios at {scenarios_dir}")

    # 2. Audit (Rollout + Judgment). Native .eval log -> per-model/behavior dir.
    log_dir = RESULTS_ROOT / args.model / args.behavior
    log_dir.mkdir(parents=True, exist_ok=True)
    _run([
        "inspect", "eval", "petri_bloom/bloom_audit",
        "-T", f"behavior={behavior_dir}",
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
