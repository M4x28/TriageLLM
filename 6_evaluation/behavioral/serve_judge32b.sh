#!/usr/bin/env bash
# Serve Qwen3-32B as the Bloom auditor/judge across two non-NVLinked L40S
# (tensor-parallel 2). NCCL_P2P_DISABLE avoids the PCIe-P2P handshake hang seen
# on non-adjacent GPU pairs; enforce-eager skips the slow compile/cudagraph step.
# GPUs passed as $1 (default "2,5"), port $2 (default 8002).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-2,5}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/data2/lbirardi/.cache/huggingface
cd /data2/lbirardi/TriageLLM
exec .venv-etl/bin/vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --port "${2:-8002}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
