#!/usr/bin/env bash
# Serve a model behind a vLLM OpenAI-compatible endpoint for behavioral eval.
#
# Used for BOTH roles:
#   * target  — the model under evaluation (e.g. our merged qwen3-1.7b checkpoint)
#   * auditor/judge — a strong local model (e.g. qwen3-8b or qwen3.6-27b)
#
# The flags below are required by Bloom (spike-verified): tool calling needs
# --enable-auto-tool-choice + a tool-call parser, and the context window must fit
# Bloom's up-to-8192 completion tokens.
#
# Usage:
#   bash serve_model.sh <model_path_or_hf_id> <served_name> <port> <gpu_id> [parser]
# Examples:
#   bash serve_model.sh data/sft/checkpoints/qwen3-1.7b/merged qwen3-1.7b 8001 0
#   bash serve_model.sh Qwen/Qwen3-8B qwen3-8b 8002 2
set -euo pipefail

MODEL="${1:?model path or HF id required}"
SERVED_NAME="${2:?served-model-name required}"
PORT="${3:?port required}"
GPU="${4:?gpu id required}"
PARSER="${5:-hermes}"                 # Qwen family uses the hermes tool-call parser
MAX_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.75}"

echo "[serve] model=${MODEL} name=${SERVED_NAME} port=${PORT} gpu=${GPU} parser=${PARSER}"

CUDA_VISIBLE_DEVICES="${GPU}" vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${PARSER}"
