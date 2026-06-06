"""Run circuit-tracer attribution on the BASE Qwen3-1.7B (transcoders valid).

Thin wrapper over the `circuit-tracer attribute` CLI. Renders each trace prompt
with the base chat template and produces an attribution graph. For `esi_leak_resp`
it first runs TWO pilot threshold settings so you can pick before doing the rest.

Run in `.venv-circuit` (circuit-tracer + nnsight). Backend nnsight is required for
Qwen3 (TransformerLens does not support it). Graphs are heavy -> gitignored.

Usage (server):
  # pilot thresholds on the leak prompt
  python interpretability/trace_base.py --pilot
  # full set with chosen thresholds, then inspect one with --server
  python interpretability/trace_base.py --node-threshold 0.5 --edge-threshold 0.9
  python interpretability/trace_base.py --only esi_leak_resp --server
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

import cases as P

# Console script lives in the active venv's bin (not on the inherited PATH when
# this script is launched via the venv python).
CT = str(Path(sys.executable).parent / "circuit-tracer")
BASE = "Qwen/Qwen3-1.7B"
TRANSCODERS = "mwhanna/qwen3-1.7b-transcoders-lowl0"
REPO = Path(__file__).resolve().parent.parent
GRAPHS = REPO / "data/eval/interp/graphs"


def rendered(tok, pid: str) -> str:
    return tok.apply_chat_template(P.chat_messages(pid), tokenize=False,
                                   add_generation_prompt=True)


def attribute(tok, pid: str, node_t: float, edge_t: float, slug: str,
              server: bool) -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    cmd = [
        CT, "attribute",
        "-m", BASE, "-t", TRANSCODERS,
        "-p", rendered(tok, pid),
        "--slug", slug,
        "--graph_file_dir", str(GRAPHS),
        "--graph_output_path", str(GRAPHS / f"{slug}.pt"),
        "--offload", "cpu",
        "--node_threshold", str(node_t),
        "--edge_threshold", str(edge_t),
    ]
    if server:
        cmd += ["--server", "--port", "8041"]
    print("[trace_base] $", " ".join(c if len(c) < 60 else c[:57] + "..." for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="two threshold settings on esi_leak_resp only")
    ap.add_argument("--node-threshold", type=float, default=0.8)
    ap.add_argument("--edge-threshold", type=float, default=0.98)
    ap.add_argument("--only", default=None, help="comma list of prompt ids")
    ap.add_argument("--server", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)

    if args.pilot:
        for nt, et in ((0.8, 0.98), (0.5, 0.9)):
            slug = f"esi_leak_resp_n{nt}_e{et}".replace(".", "")
            attribute(tok, "esi_leak_resp", nt, et, slug, server=False)
        print("[trace_base] pilots done — compare graphs, then re-run with chosen "
              "--node-threshold/--edge-threshold")
        return 0

    ids = (args.only.split(",") if args.only else list(P.PROMPTS))
    for pid in ids:
        slug = f"{pid}_n{args.node_threshold}_e{args.edge_threshold}".replace(".", "")
        attribute(tok, pid, args.node_threshold, args.edge_threshold, slug,
                  server=args.server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
