# Phase2 16K Internal-State Comparison 2026-05-04

- Harness-state run: `/home/panzihang/src/LMCache-phase0-codex/phase2_runs/20260504_16k16384_qwen7b_harness_state_control`
- Internal-state run: `/home/panzihang/src/LMCache-phase0-codex/phase2_runs/20260504_16k16384_qwen7b_internal_state_compare`

| scenario | harness mean_subsequent_ttft_ms | internal mean_subsequent_ttft_ms | delta % | harness match | internal match | harness final | internal final |
|---|---:|---:|---:|---:|---:|---|---|
| no_promotion | 315.822 | 359.832 | 13.935 | 1.0 | 1.0 | BASE_READY | BASE_READY |
| promotion | 120.45 | 128.889 | 7.006 | 1.0 | 1.0 | FULL_READY | FULL_READY |
| oracle_full_ready | 120.357 | 127.417 | 5.866 | 1.0 | 1.0 | FULL_READY | FULL_READY |

## Evidence

- Internal raw log PHASE2_INTERNAL_STATE_* lines: 80
- Internal raw log PHASE2_CORE_AUTO_DECISION lines: 36
- Harness raw log PHASE2_CORE_AUTO_DECISION lines: 36

## Interpretation

- The comparison checks whether the new internal state source reproduces the earlier harness-state behavior under the same 16K settings.
- A delta within roughly 5-10% supports using the harness-state results as trustworthy evidence while claiming the internal state source is now wired into LMCache.
- Promotion remains request-driven; this comparison does not implement an independent background scheduler.
