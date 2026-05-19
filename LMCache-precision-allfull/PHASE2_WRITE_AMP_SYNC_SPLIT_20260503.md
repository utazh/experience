# Phase 2 Write Amplification Sync Split (2026-05-03)

Dual run: `phase2_runs/20260424_163955_32k32760_dualwrite_qwen7b_matrix`

Single-write baseline: `phase2_runs/20260424_164414_32k32760_singlewrite_qwen7b_baseline`

| metric | value |
| --- | ---: |
| single_prime_ttft_ms | 10839.502 |
| dual_prime_ttft_ms | 11480.835 |
| ttft_overhead_ms | 641.333 |
| ttft_overhead_pct | 5.917 |
| single_store_ms_total_logged | 1003.1898000000001 |
| dual_store_ms_total_logged | 1635.176 |
| store_ms_overhead_logged | 631.986 |
| dual_write_sync_submit_ms | 1428.344 |
| dual_write_base_encode_ms | 1418.4621000000002 |
| dual_write_full_submit_ms | 8.971599999999999 |
| dual_write_base_put_ms | 0.9107000000000001 |

## Interpretation
- `ttft_overhead_ms` is the user-visible wall-clock delta between the dual-write run and the single-write run.
- `dual_write_base_encode_ms` is a sum across 16 chunked store callbacks. It measures accumulated foreground CPU work inside LMCache, not an independently measured critical-path delta versus the single-write run.
- The apparent mismatch (`base_encode=1418.462 ms` vs `TTFT overhead=641.333 ms`) is expected once the denominator is corrected: the single-write baseline already pays `single_store_ms_total_logged=1003.1898 ms`, while dual-write pays `dual_store_ms_total_logged=1635.176 ms`. The logged store overhead is therefore `631.986 ms`, which closely matches the TTFT overhead.
- `full_submit_ms=8.9716 ms` remains small; the first-request cost is dominated by base encode and existing foreground store/offload work, not by submitting full to disk.
- See `PHASE2_WRITE_AMP_TIMING_CLARIFICATION_20260503.md/json` for the parser-backed clarification and standalone CPU int8 encode microbench.
