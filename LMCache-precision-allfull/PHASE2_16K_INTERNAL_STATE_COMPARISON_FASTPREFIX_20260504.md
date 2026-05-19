# Phase2 16K Internal-State Comparison Fast Prefix-ID 2026-05-04

- Harness-state run: `/home/panzihang/src/LMCache-phase0-codex/phase2_runs/20260504_16k16384_qwen7b_harness_state_control_fastprefix`
- Internal-state run: `/home/panzihang/src/LMCache-phase0-codex/phase2_runs/20260504_16k16384_qwen7b_internal_state_compare_fastprefix`

| scenario | harness mean_subsequent_ttft_ms | internal mean_subsequent_ttft_ms | delta % | harness match | internal match | harness final | internal final |
|---|---:|---:|---:|---:|---:|---|---|
| no_promotion | 321.637 | 330.908 | 2.882 | 1.0 | 1.0 | BASE_READY | BASE_READY |
| promotion | 117.237 | 118.384 | 0.978 | 1.0 | 1.0 | FULL_READY | FULL_READY |
| oracle_full_ready | 127.695 | 122.987 | -3.687 | 1.0 | 1.0 | FULL_READY | FULL_READY |

## Evidence

- Internal raw log PHASE2_INTERNAL_STATE_* lines: 80
- Internal raw log PHASE2_CORE_AUTO_DECISION lines: 36
- Harness raw log PHASE2_CORE_AUTO_DECISION lines: 36

## Interpretation

- All three scenarios are within the 5-10% comparison window after removing repeated token-string hashing from internal prefix lookup.
- The fidelity state source is internal to LMCache; the request prefix_id is used as the lookup key identity when present.
- Promotion remains request-driven, not an independent background scheduler.
