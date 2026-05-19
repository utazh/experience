# Phase 1/4 Environment Notes

Updated: 2026-05-18

## Server

- Host: 10.147.20.12
- User workspace: /home/panzihang/src/experience
- Python environment: /home/panzihang/venvs/vllm-stable
- GPUs: 4x NVIDIA GeForce RTX 3090
- Model used in Phase 1/4 smoke benchmark: /data1/llm/Qwen/Qwen2-1.5B
- Served model name: qwen2-1.5b

## LMCache/vLLM paths

- Official/stable server baseline: /home/panzihang/src/LMCache-phase0-codex
- Uploaded source tree: /home/panzihang/src/experience/LMCache-dev
- Precision experiment server copy: /home/panzihang/src/experience/LMCache-precision-allfull
- Benchmark client source: /home/panzihang/src/experience/LMCache-dev

## Compatibility constraints

- The uploaded newer LMCache-dev Python server path is not usable with the currently installed compiled lmcache.c_ops extension.
- Observed failure: missing GPUKVFormat enum member `NL_X_TWO_NB_NH_BS_HS`.
- `nvcc` is not available in the current shell, so rebuilding CUDA/c_ops is not available yet.
- Current Phase 4 all-base namespace work is pure Python and does not require rebuilding c_ops.
- Future Phase 8 XNOR+popcount should start with a pure Python/CPU prototype or prepare a compatible CUDA toolchain/wheel first.

## Phase 1 result files

- /home/panzihang/src/experience/runs/phase1_synthetic_long_doc_qa_comparison.csv
- /home/panzihang/src/experience/runs/phase1_synthetic_long_doc_qa_comparison.json

## Phase 4 interpretation note

`all-base` in the current precision experiment copy validates namespace routing only. It stores/retrieves the same full KV tensor shape and dtype under `lmcache.tag.precision=base`; it is not yet a real compressed/base KV storage path. Therefore, all-base should be compared for functional overhead and key isolation, not expected byte-transfer savings.
