"""Shared helpers for TriageLLM step 7 deploy (GGUF Q4_K_M).

Reuses by importlib (no duplication):
  * step 6 common: path const + ModelSpec / MODELS registry + setup_logging
  * step 6 prompts: SYSTEM_PROMPT + IDENTITY_PROBES + parsers + keywords

Adds locally:
  * deploy-specific paths: DEPLOY_DIR / GGUF_DIR / EVAL_Q4KM_DIR
  * load_gguf_model(path, n_ctx, n_threads) -> llama_cpp.Llama
  * generate_gguf(llama, system, user, max_new_tokens) mirror of
    `6_evaluation.common.generate` but for the GGUF backend
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEPLOY_DIR = DATA_DIR / "deploy"
GGUF_DIR = DEPLOY_DIR / "gguf"
EVAL_Q4KM_DIR = DEPLOY_DIR / "eval_q4km"
COMPARE_FILE = DEPLOY_DIR / "compare_bf16_q4km.json"
POCKETPAL_SAMPLES_FILE = DEPLOY_DIR / "pocketpal_samples.jsonl"


def _load_sibling_module(folder_glob: str, file_name: str, name: str):
    """Load a Python module from a sibling step folder by glob."""
    candidates = sorted(REPO_ROOT.glob(folder_glob))
    if not candidates:
        raise FileNotFoundError(f"no sibling '{folder_glob}' folder under {REPO_ROOT}")
    path = candidates[-1] / file_name
    spec = _ilu.spec_from_file_location(name, str(path))
    module = _ilu.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Step 6 modules: registry, paths, prompts. Reused verbatim.
_step6_common = _load_sibling_module("6_evaluation", "common.py",
                                     "triagellm_step6_common")
_step6_prompts = _load_sibling_module("6_evaluation", "prompts.py",
                                      "triagellm_step6_prompts")

# Re-exports from step 6.
ModelSpec = _step6_common.ModelSpec
MODELS = _step6_common.MODELS
setup_logging = _step6_common.setup_logging
load_jsonl = _step6_common.load_jsonl
write_jsonl = _step6_common.write_jsonl
AUGMENT_DIR = _step6_common.AUGMENT_DIR
INTERNAL_DIR = _step6_common.INTERNAL_DIR
CHECKPOINT_DIR = _step6_common.CHECKPOINT_DIR

SYSTEM_PROMPT = _step6_prompts.SYSTEM_PROMPT
IDENTITY_PROBES = _step6_prompts.IDENTITY_PROBES
PEDIATRIC_CONTENT_KEYWORDS = _step6_prompts.PEDIATRIC_CONTENT_KEYWORDS
extract_triage_labels = _step6_prompts.extract_triage_labels
is_refusal = _step6_prompts.is_refusal


def load_gguf_model(gguf_path: Path, *,
                    n_ctx: int = 4096,
                    n_threads: int | None = None,
                    n_gpu_layers: int = 0,
                    logits_all: bool = False,
                    verbose: bool = False) -> Any:
    """Load a GGUF model via llama-cpp-python.

    Defaults to CPU-only inference (`n_gpu_layers=0`) to approximate the
    ARM Neon path used on the Android device. `n_threads=None` lets
    llama.cpp pick a sensible default based on the host CPU.

    `logits_all=True` is required to compute per-token logprobs (used for
    the validation NLL). Slows generation modestly; the latency numbers
    are still informative for ranking.
    """
    from llama_cpp import Llama

    llama = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        logits_all=logits_all,
        verbose=verbose,
        chat_format=None,  # use the chat template embedded in the GGUF metadata
    )
    return llama


def generate_gguf(llama, system: str, user: str,
                  max_new_tokens: int = 1024) -> tuple[str, float, int]:
    """Greedy generation through `create_chat_completion`.

    Returns `(decoded_text, elapsed_seconds, completion_tokens)`. The token
    count is reported by llama.cpp itself, so latency in tokens/s can be
    derived as `completion_tokens / elapsed`.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    t0 = time.time()
    out = llama.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,  # deterministic greedy decoding
    )
    elapsed = float(time.time() - t0)
    text = out["choices"][0]["message"]["content"] or ""
    completion_tokens = int(out.get("usage", {}).get("completion_tokens", 0))
    return text.strip(), elapsed, max(completion_tokens, 1)


def tokenize_gguf(llama, text: str) -> list[int]:
    """Return the token id list produced by the GGUF tokenizer."""
    return llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)


def compute_nll_gguf(llama, prompt: str) -> tuple[float, int]:
    """Token-level mean NLL over `prompt` under the GGUF model.

    llama.cpp exposes `logits_all=True` evaluation through `eval` + the
    `scores()` accessor. We use `__call__(prompt, max_tokens=0,
    logprobs=...)` which returns the per-token log-probabilities for the
    input sequence. Returns `(mean_nll, n_tokens)`.
    """
    import math

    # logprobs=0 returns the per-token logprob of the actual token only.
    completion = llama(prompt, max_tokens=0, logprobs=0, echo=True)
    token_logprobs = completion["choices"][0]["logprobs"]["token_logprobs"]
    # Drop the first token (no preceding context -> logprob is None).
    valid = [lp for lp in token_logprobs if lp is not None]
    if not valid:
        return float("inf"), 0
    mean_nll = -sum(valid) / len(valid)
    return mean_nll, len(valid)


__all__ = [
    "REPO_ROOT", "DATA_DIR", "DEPLOY_DIR", "GGUF_DIR", "EVAL_Q4KM_DIR",
    "COMPARE_FILE", "POCKETPAL_SAMPLES_FILE",
    "AUGMENT_DIR", "INTERNAL_DIR", "CHECKPOINT_DIR",
    "ModelSpec", "MODELS",
    "setup_logging", "load_jsonl", "write_jsonl",
    "SYSTEM_PROMPT", "IDENTITY_PROBES", "PEDIATRIC_CONTENT_KEYWORDS",
    "extract_triage_labels", "is_refusal",
    "load_gguf_model", "generate_gguf", "tokenize_gguf", "compute_nll_gguf",
]
