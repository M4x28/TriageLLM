"""Run a Petri open-ended behavioral audit against the local triage target.

Thin orchestration only - it does not reimplement Petri. It (1) reuses the
behavioral endpoint config, (2) pins the deployed triage SYSTEM_PROMPT on the
target via a custom target agent (see `triage_target.py`; Petri has no native
`target_system_prompt` param), (3) builds the judge dimension set (our custom
clinical dimensions + a few built-ins), and (4) runs `inspect_petri.audit`
through the inspect_ai Python API (required to pass a custom `target=` agent),
writing native inspect `.eval` logs under `data/eval/petri/<model>/`.

Because a custom target Agent can only be passed in Python, this uses the
`inspect_ai.eval()` API rather than the `inspect eval` CLI. Run in the same venv
as Bloom (`.venv-bloom`) with `inspect-petri` installed.

Prereqs: target + auditor/judge served via the behavioral `serve_*.sh` scripts.

Usage:
  # full seed set
  python 6_evaluation/petri/run_petri.py --model qwen3-1.7b
  # pilot subset
  python 6_evaluation/petri/run_petri.py --model qwen3-1.7b \
      --seeds caregiver_resists_referral,respiratory_danger_signs \
      --max-turns 10 --epochs 3
  # print the resolved config without running
  python 6_evaluation/petri/run_petri.py --model qwen3-1.7b --seeds ... --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEHAVIORAL = HERE.parent / "behavioral"
SEEDS_DIR = HERE / "seeds"
JUDGE_DIMS_DIR = HERE / "judge_dimensions"
WORK_DIR = HERE / ".work"                 # gitignored seed-subset staging
REPO_ROOT = HERE.parent.parent
RESULTS_ROOT = REPO_ROOT / "data" / "eval" / "petri"

# Reuse the behavioral endpoint config (one source for local<->API swap).
sys.path.insert(0, str(BEHAVIORAL))
import endpoints  # noqa: E402

# Built-in Petri judge dimensions added alongside our clinical ones.
BUILTIN_DIMENSIONS = ["concerning", "eval_awareness", "scenario_realism"]


def _resolve_seeds(selection: str | None) -> Path:
    """Return a directory of `.md` seeds: the full set, or a staged subset.

    Petri reads a directory flat (`dir.glob('*.md')`); a subset is staged as
    copies under `.work/<sel>/` so sample ids (filename stems) are preserved.
    """
    if not selection:
        return SEEDS_DIR
    names = [s.strip() for s in selection.split(",") if s.strip()]
    missing = [n for n in names if not (SEEDS_DIR / f"{n}.md").exists()]
    if missing:
        raise SystemExit(f"ERROR: unknown seed(s): {missing}. "
                         f"Available: {sorted(p.stem for p in SEEDS_DIR.glob('*.md'))}")
    subset = WORK_DIR / ("_".join(names)[:60] or "subset")
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True)
    for n in names:
        shutil.copy2(SEEDS_DIR / f"{n}.md", subset / f"{n}.md")
    return subset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="target served-model-name (e.g. qwen3-1.7b)")
    ap.add_argument("--seeds", default=None,
                    help="comma list of seed ids (filename stems); default = all")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3,
                    help="rollouts per seed")
    ap.add_argument("--auditor-model", default=endpoints.DEFAULT_AUDITOR.model)
    ap.add_argument("--target-port", type=int, default=endpoints.DEFAULT_TARGET.port)
    ap.add_argument("--auditor-port", type=int, default=endpoints.DEFAULT_AUDITOR.port)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved eval config and exit")
    args = ap.parse_args()

    seeds_path = _resolve_seeds(args.seeds)

    # Endpoints: target = model under test; auditor + judge share one strong
    # local model (Qwen3-32B by default via --auditor-model).
    target = endpoints.Endpoint("target", args.model, args.target_port)
    auditor = endpoints.Endpoint("auditor", args.auditor_model, args.auditor_port)
    judge = endpoints.Endpoint("judge", args.auditor_model, args.auditor_port)
    roles = endpoints.configure_local(target=target, auditor=auditor, judge=judge)

    # Judge dimensions: our clinical set (directory) + selected built-ins.
    from inspect_petri import judge_dimensions  # noqa: E402
    dimensions = list(judge_dimensions(str(JUDGE_DIMS_DIR))) + BUILTIN_DIMENSIONS
    dim_names = [d if isinstance(d, str) else d.name for d in dimensions]

    log_dir = RESULTS_ROOT / args.model
    seed_ids = sorted(p.stem for p in seeds_path.glob("*.md"))

    if args.dry_run:
        from triage_target import SYSTEM_PROMPT  # noqa: E402
        print("[run_petri] DRY RUN — resolved config")
        print(f"  task            : inspect_petri.audit")
        print(f"  seeds dir       : {seeds_path}")
        print(f"  seeds           : {seed_ids}")
        print(f"  max_turns       : {args.max_turns}")
        print(f"  epochs          : {args.epochs}")
        print(f"  target_tools    : none")
        print(f"  judge dimensions: {dim_names}")
        print(f"  model_roles     : auditor={roles['auditor']} "
              f"target={roles['target']} judge={roles['judge']}")
        print(f"  log_dir         : {log_dir}")
        print(f"  target sysprompt: pinned via triage_target_agent() -> "
              f"{SYSTEM_PROMPT.splitlines()[0][:70]}...")
        return 0

    # Real run: inspect_ai Python API (needed to pass the custom target agent).
    from inspect_ai import eval as inspect_eval
    from inspect_petri import audit
    from triage_target import triage_target_agent

    log_dir.mkdir(parents=True, exist_ok=True)
    task = audit(
        seed_instructions=str(seeds_path),
        max_turns=args.max_turns,
        judge_dimensions=dimensions,
        target=triage_target_agent(),
        target_tools="none",     # deployed triage target uses no tools
    )
    inspect_eval(
        task,
        model_roles={
            "auditor": roles["auditor"],
            "target": roles["target"],
            "judge": roles["judge"],
        },
        log_dir=str(log_dir),
        epochs=args.epochs,
    )
    print(f"[run_petri] done. Native logs in {log_dir}")
    print(f"[run_petri] view: inspect view --log-dir {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
