# Phase 2 Write Amplification Timing Clarification

Dual log: `phase2_runs/20260424_163955_32k32760_dualwrite_qwen7b_matrix/run.log`

Single log: `phase2_runs/20260424_164414_32k32760_singlewrite_qwen7b_baseline/run.log`

## Log-Derived Comparison

| metric | value |
| --- | ---: |
| dual_ttft_ms | 11480.835 |
| single_ttft_ms | 10839.502 |
| ttft_overhead_ms | 641.333 |
| dual_e2e_ms | 11530.695 |
| single_e2e_ms | 10889.779 |
| e2e_overhead_ms | 640.916 |
| dual_store_cost_ms_sum | 1635.176 |
| single_store_cost_ms_sum | 1003.19 |
| store_cost_overhead_ms_sum | 631.986 |
| dual_store_log_wall_span_ms | 10921.0 |
| single_store_log_wall_span_ms | 10283.0 |
| dual_write_events | 16 |
| dual_write_chunks | 127 |
| dual_write_base_encode_ms_sum | 1418.462 |
| dual_write_full_submit_ms_sum | 8.972 |
| dual_write_base_put_ms_sum | 0.911 |
| dual_write_sync_ms_sum | 1428.344 |
| dual_write_log_wall_span_ms | 10922.0 |
| dual_write_base_locations | LocalCPUBackend |
| dual_write_full_locations | LocalDiskBackend |

## Standalone CPU Int8 Microbench

| metric | value |
| --- | ---: |
| model | /data1/llm/Qwen/Qwen2.5-7B-Instruct |
| shape | {'layers': 28, 'kv_heads': 4, 'head_dim': 128, 'chunk_tokens': 2048} |
| dtype | bfloat16 |
| input_bytes_per_chunk | 117440512 |
| encoded_bytes_per_chunk | 60555264 |
| compression_ratio_per_chunk | 0.515625 |
| chunks_for_estimate | 16 |
| repeats | 1 |
| encode_ms_per_chunk_mean | 161.39 |
| encode_ms_per_chunk_p50 | 161.39 |
| encode_plus_copy_ms_per_chunk_mean | 182.957 |
| encode_plus_copy_ms_per_chunk_p50 | 182.957 |
| estimated_encode_ms_total | 2582.235 |
| estimated_encode_plus_copy_ms_total | 2927.316 |

## Interpretation
- ttft_overhead_ms is a wall-clock delta between two complete requests/runs; it is the user-visible first-request cost.
- dual_write_base_encode_ms_sum is a sum of per-chunk CPU encode sections inside LMCache; it measures accumulated foreground work, not an independently measured critical-path delta versus the single-write run.
- single-write already pays prefill, D2H/offload, and base store costs; therefore dual_write_base_encode_ms_sum can be larger than ttft_overhead_ms without contradiction.
- Use TTFT overhead for user-visible latency, and report base_encode/full_submit/base_put sums as a decomposition of LMCache foreground work.
