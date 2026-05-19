#!/usr/bin/env bash
set -euo pipefail
source ~/venvs/vllm-stable/bin/activate
export VLLM_USE_V1=1
export LMCACHE_CONFIG_FILE=/home/panzihang/src/LMCache-phase0-codex/phase0_smoke.yaml
export CUDA_VISIBLE_DEVICES=0
vllm serve /home/panzihang/models/Qwen2.5-0.5B-Instruct   --served-model-name Qwen2.5-0.5B-Instruct   --port 8010   --gpu-memory-utilization 0.5   --enforce-eager   --trust-remote-code   --kv-transfer-config kv_connector:LMCacheConnectorV1
