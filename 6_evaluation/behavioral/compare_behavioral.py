"""Cross-model comparison for one behavior, read from native inspect .eval logs.

No custom schema: this reads the Bloom/inspect_ai eval logs written under
`data/eval/behavioral/<model>/<behavior>/` and prints a per-model table of the
scored dimension means (the behavior score plus Bloom's eval_awareness and
scenario_realism). Run inside `.venv-bloom`.

Usage:
  python 6_evaluation/behavioral/compare_behavioral.py --behavior false_reassurance
"""
from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parent.parent / "data" / "eval" / "behavioral"


def _latest_log(model_behavior_dir: Path) -> str | None:
    """Most recent .eval log path under a model/behavior dir, or None."""
    logs = list_eval_logs(str(model_behavior_dir))
    if not logs:
        return None
    # list_eval_logs returns newest-last by mtime-sortable name; take the last.
    return sorted(l.name for l in logs)[-1]


def _dimension_means(log_path: str) -> tuple[dict[str, float], int]:
    """Return {dimension_name: mean} and sample count from an eval log."""
    log = read_eval_log(log_path)
    means: dict[str, float] = {}
    scores = (log.results.scores if log.results else []) or []
    for sc in scores:
        metric = sc.metrics.get("mean")
        if metric is not None:
            means[sc.name] = round(float(metric.value), 3)
    n = log.results.total_samples if log.results else 0
    return means, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True)
    args = ap.parse_args()

    rows: list[tuple[str, int, dict[str, float]]] = []
    all_dims: list[str] = []
    if RESULTS_ROOT.exists():
        for model_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
            mb = model_dir / args.behavior
            if not mb.exists():
                continue
            log_path = _latest_log(mb)
            if not log_path:
                continue
            means, n = _dimension_means(log_path)
            rows.append((model_dir.name, n, means))
            for d in means:
                if d not in all_dims:
                    all_dims.append(d)

    if not rows:
        print(f"No behavioral logs found for '{args.behavior}' under {RESULTS_ROOT}")
        return 1

    # Put the behavior's own dimension first if present.
    all_dims.sort(key=lambda d: (d != args.behavior, d))
    header = ["model", "n"] + all_dims
    widths = [max(len(header[0]), *(len(r[0]) for r in rows)), 4] + \
             [max(len(d), 6) for d in all_dims]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(f"\nBehavior: {args.behavior}  (dimension means, native inspect logs)\n")
    print(fmt(header))
    print(fmt(["-" * w for w in widths]))
    for model, n, means in rows:
        cells = [model, str(n)] + [
            (f"{means[d]:.3f}" if d in means else "-") for d in all_dims
        ]
        print(fmt(cells))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
