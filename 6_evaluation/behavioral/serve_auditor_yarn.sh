#!/usr/bin/env bash
# Serve the Bloom auditor/judge (Qwen3-8B) with YaRN rope-scaling to 65k context.
# Needed because verbose behavioral rollouts (e.g. convulsion cases) overflow the
# native 32k window during judgment. GPU is passed as $1 (default 2), port $2 (8002).
set -euo pipefail
GPU="${1:-2}"
PORT="${2:-8002}"
cd /data2/lbirardi/TriageLLM
HF_HOME=/data2/lbirardi/.cache/huggingface CUDA_VISIBLE_DEVICES="${GPU}" \
  .venv-etl/bin/vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --port "${PORT}" \
  --gpu-memory-utilization 0.80 \
  --max-model-len 65536 \
  --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
