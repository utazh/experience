# Phase2 16K No-Promotion Fast Prefix-ID Check 2026-05-04

- Harness reference: `/home/panzihang/src/LMCache-phase0-codex/phase2_runs/20260504_16k16384_qwen7b_harness_state_nopromo_r2`
- Internal fast-prefix run: `/home/panzihang/src/LMCache-phase0-codex/phase2_runs/20260504_16k16384_qwen7b_internal_state_nopromo_r3_fastprefix`

| metric | harness | internal | delta % |
|---|---:|---:|---:|
| mean_subsequent_ttft_ms | 333.844 | 334.592 | 0.224 |
| mean_followup_match_rate | 1.0 | 1.0 |  |

## Note

- Internal state lookup uses the request prefix_id as the state-store key when available; this removes repeated 16K token string hashing from the measured followup path.
- The prefix_id is an identity hint, not the fidelity state source. The state still comes from LMCache internal state_store.
