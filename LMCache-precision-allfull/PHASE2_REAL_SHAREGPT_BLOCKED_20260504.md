# Phase2 Real ShareGPT Blocked 2026-05-04

The real-dataset runner supports ShareGPT via:

```bash
benchmarks/quality_aware_kv/phase2/run_phase2_real_dataset.py \
  --dataset sharegpt \
  --dataset-path /data1/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json
```

Current blocker:

- No non-empty ShareGPT JSON file exists under `/data1/datasets`, `/home/panzihang/src`, or the HuggingFace cache on the server.
- The server cannot reach HuggingFace from this environment. `hf_hub_download()` failed with `Network is unreachable` while requesting `https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/...`.
- I did not substitute another dataset or synthetic conversation text, because that would weaken the claim that this is a real ShareGPT validation.

What is ready:

- The runner can load ShareGPT's standard `conversations` JSON schema.
- It constructs hot-prefix prompts from real multi-turn conversation transcripts.
- It runs internal-state mode only; no `lmcache.state_store_v0.state` is passed by the harness.
- It emits standard `PHASE2_*` logs and can be parsed by `parse_phase2_hot_prefix.py`.

Next action once the file is available:

```bash
cd /home/panzihang/src/LMCache-phase0-codex
RUN_DIR=phase2_runs/20260504_real_sharegpt_16k_qwen7b_internal
CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 LMCACHE_CLEANUP_BASE_ON_FULL_STORE=True \
PYTHONPATH=/home/panzihang/src/LMCache-phase0-codex:${PYTHONPATH:-} \
LD_LIBRARY_PATH=/home/panzihang/venvs/vllm-stable/lib/python3.10/site-packages/torch/lib:/home/panzihang/venvs/vllm-stable/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
/home/panzihang/venvs/vllm-stable/bin/python benchmarks/quality_aware_kv/phase2/run_phase2_real_dataset.py \
  --model /data1/llm/Qwen/Qwen2.5-7B-Instruct \
  --dataset sharegpt \
  --dataset-path /data1/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --context-len 16384 \
  --num-prefixes 1 \
  --run-scenarios no_promotion promotion oracle_full_ready \
  --num-followup-requests 6 \
  --promotion-min-access-count 1 \
  --max-tokens 16 \
  --post-request-settle-ms 250 \
  --gpu-memory-utilization 0.80 \
  --max-local-cpu-size 10 \
  --local-disk "$RUN_DIR/local_disk" \
  --max-local-disk-size 80 \
  > "$RUN_DIR/run.log" 2>&1
```
