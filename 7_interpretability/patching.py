"""Activation patching — DEFERRED (run only if token-prob + logit-lens are not
already conclusive).

Plan (when run):
  PRIMARY = within-model, constant-input minimal pair. Take `esi_leak_resp` and a
  NATURALLY clean IMCI generation of the SAME prompt (from generate_outputs.py /
  outputs.json — never a steered prompt). Patch the clean run's residual-stream /
  MLP-output into the leaked run, per layer and position, and measure the change
  in the ESI-IMCI margin to localize the causal layers of the leak.

  EXPLORATORY ONLY = cross-model (round-5 activations into round-3). Representations
  shift with fine-tuning, so treat any result as a hint, not proof.

Intentionally left unimplemented until the earlier, cheaper measures justify it,
to avoid a costly analysis that is easy to misread.
"""
raise SystemExit("patching.py is deferred — run generate_outputs.py + "
                 "finetune_diff.py first; implement patching only if needed.")
