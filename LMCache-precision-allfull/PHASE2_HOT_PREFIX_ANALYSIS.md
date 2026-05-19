# Phase 2 Hot Prefix Analysis

## Scope
Initial Phase 2 experiments ran on the remote server only. The local workspace was not modified.

Runtime:
- model: `/data1/llm/Qwen/Qwen2.5-7B-Instruct`
- vLLM: `0.19.0`
- LMCache native `c_ops`: enabled
- GPU: `CUDA_VISIBLE_DEVICES=2`
- layerwise: enabled
- promotion mode: `full_materialization_proxy`

## Run Directories
- 1K smoke: `phase2_runs/20260423_180307_promotion_smoke_1k_retry`
- 4K matrix: `phase2_runs/20260423_180420_hot_prefix_4k_initial_matrix`
- 16K matrix: `phase2_runs/20260423_180603_hot_prefix_16k_initial_matrix`
- failed smoke attempt: `phase2_runs/20260423_180216_promotion_smoke_1k` failed before requests because `gpu_memory_utilization=0.55` left no KV cache memory.

## Result Summary
| context | scenario | final_state | first_followup_ttft_ms | mean_subsequent_ttft_ms | promotion_success | full_reuse_count | promotion_reuse_rate | match_rate |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | promotion | FULL_READY | 71.161 | 45.167 | 1 | 1 | 1.0 | 1.0 |
| 4096 | no_promotion | BASE_READY | 130.409 | 127.789 | 0 | 0 |  | 1.0 |
| 4096 | promotion | FULL_READY | 153.357 | 59.282 | 1 | 2 | 2.0 | 1.0 |
| 4096 | oracle_full_ready | FULL_READY | 60.952 | 57.706 | 0 | 0 |  | 1.0 |
| 16384 | no_promotion | BASE_READY | 449.334 | 448.788 | 0 | 0 |  | 1.0 |
| 16384 | promotion | FULL_READY | 425.453 | 111.102 | 1 | 2 | 2.0 | 1.0 |
| 16384 | oracle_full_ready | FULL_READY | 109.169 | 120.952 | 0 | 0 |  | 1.0 |

## Main Observations
- The state machine is working in the harness: `BASE_READY -> PROMOTING -> FULL_READY` was observed in both the 1K smoke and the 4K/16K matrices.
- At 4K, no-promotion subsequent TTFT was `127.789` ms, while promotion subsequent TTFT was `59.282` ms. The promotion path was within measurement noise of oracle full-ready (`57.706` ms).
- At 16K, no-promotion subsequent TTFT was `448.788` ms, while promotion subsequent TTFT was `111.102` ms. Oracle full-ready was `120.952` ms.
- Output match stayed at `1.0` for all measured follow-up requests.
- Full-ready follow-up retrieval is faster despite larger bytes. At 16K, base follow-up retrieve was about `384.187` ms for `0.4409` GB, while full follow-up retrieve in the promotion scenario was about `48.3896` ms for `0.875` GB. This is consistent with Phase 1: current base path pays heavy int8 decode cost.

## Interpretation
These initial experiments validate the Phase 2 hypothesis at the benchmark/state-machine level: once a hot prefix reaches `FULL_READY`, later requests behave like the oracle full-ready upper bound and avoid the slower base decode path.

The initial 2026-04-23 result should not be reported as final background I/O promotion: those runs used a full-materialization proxy because the first-request dual-store path did not exist yet. The 2026-04-24 update below adds real first-request base/full dual-store and uses full data already stored in the slow tier, but promotion is still request-driven rather than an independent background scheduler.

## 2026-04-24 Dual-Write Update
A real first-request dual-store path was added on the remote server. New results are recorded in:
- `PHASE2_DUAL_WRITE_RESULTS_20260424.md`
- `PHASE2_DUAL_WRITE_RESULTS_20260424.json`
- `PHASE2_DUAL_WRITE_RESULTS_20260424.csv`

Key 32K small-model validation result (`/data1/llm/Llama-3.2-1B`): no-promotion subsequent TTFT `245.735` ms, promotion subsequent TTFT `101.579` ms, oracle `105.514` ms, reuse rate `5.0`, match rate `1.0`. This validated the dual-write/full-disk retrieve-writeback path before the final Qwen2.5-7B run below.

## 2026-04-24 Qwen2.5-7B Dual-Write Update
Qwen2.5-7B 32K-class results are now available in `PHASE2_DUAL_WRITE_RESULTS_20260424.md`. Because Qwen2.5-7B has `max_position_embeddings=32768`, the successful run used `context_len=32760`, `max_model_len=32768`, and `max_tokens=4`.

Main result: no-promotion subsequent TTFT `541.3` ms, promotion subsequent TTFT `313.401` ms, oracle `324.729` ms. Promotion improved subsequent TTFT by `227.899` ms (`42.102`%), with reuse rate `5.0` and match rate `1.0`.

First-request cost: single-write prime TTFT `10839.502` ms, dual-write prime TTFT `11480.835` ms, overhead `641.333` ms. Store log total increased from `1003.1898` ms to `1635.176` ms. The Qwen slow-tier full footprint is about `1.7363` GiB per 32760-token prefix; the three-prefix matrix used `5593157632` bytes and `381` files.

Boundary clarification: the current implementation is eager dual-write. Hotness controls promotion from slow tier to fast tier, but does not prevent cold-prefix full writes to slow tier. This should be reported as storage/I/O overhead unless a lazy full-write policy is added later.

