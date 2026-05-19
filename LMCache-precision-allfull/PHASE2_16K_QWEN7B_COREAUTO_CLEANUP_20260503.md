# Phase 2 16K Qwen2.5-7B Core-Auto Cleanup Results (2026-05-03)

Run dir: `phase2_runs/20260503_16k16384_qwen7b_phase2_coreauto_cleanup`

## Summary

| scenario | final_state | mean_followup_ttft_ms | mean_subsequent_ttft_ms | match_rate | full_reuse_count | promotion_success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no_promotion | BASE_READY | 416.137 | 402.071 | 1.0 | 0 | 0 |
| promotion | FULL_READY | 168.352 | 130.853 | 1.0 | 5 | 1 |
| oracle_full_ready | FULL_READY | 132.597 | 132.742 | 1.0 | 0 | 0 |

## Derived Comparisons

- Promotion vs no-promotion subsequent TTFT: `402.071 -> 130.853 ms`, delta `271.218 ms`, improvement `67.455%`.
- Promotion vs oracle subsequent TTFT: `130.853 vs 132.742 ms`, delta `-1.889 ms`.
- Promotion vs no-promotion mean followup TTFT: `416.137 -> 168.352 ms`, delta `247.785 ms`, improvement `59.544%`.

## Evidence

- Core auto decisions parsed: `36`; request rows using core auto: `18`; mismatches: `0`.
- Routing check ok: `True`; base locations: `['LocalCPUBackend']`; full locations: `['LocalDiskBackend']`; full disk files: `192`; base disk files: `0`.
- Promotion cleanup removed base chunks total: `64`.

## Boundary

- Promotion is still request-driven full-disk retrieve/writeback, not an independent background scheduler.
- The state source is still harness-provided state_store_v0 metadata; fidelity selection runs through core policy.
