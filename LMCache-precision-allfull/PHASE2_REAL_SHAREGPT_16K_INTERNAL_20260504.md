# Phase2 Real ShareGPT 16K Internal-State Results (2026-05-04)

## Setup
- Model: `/data1/llm/Qwen/Qwen2.5-7B-Instruct`
- Dataset: `/data1/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json`
- Context length: 16384 tokens
- State source: internal LMCache fidelity state store; the harness does not pass `lmcache.state_store_v0.state`.
- Promotion mode: request-driven full-disk retrieve/writeback, not an independent background scheduler.

## Dataset Slice
- row_index: `0`
- record_id: `QWJhYvA_0`
- source_ids: `['QWJhYvA_0', 'i6IyJda_0', 'A5AbcES_0', 'hRPPgZT_0', 'hRPPgZT_11', 'hRPPgZT_17', 'IWkMGRK_0', 'yn2eWCt_0']`
- source_conversation_count: `13`
- construction: Concatenated real ShareGPT conversation transcripts, head-truncated to the target context length.

## Results
| scenario | mean_subsequent_ttft_ms | mean_followup_ttft_ms | match_rate | final_state | full_reuse_count | promotion_success |
|---|---:|---:|---:|---|---:|---:|
| no_promotion | 388.808 | 387.715 | 1.000 | BASE_READY | 0 | 0 |
| promotion | 128.820 | 166.923 | 1.000 | FULL_READY | 5 | 1 |
| oracle_full_ready | 129.188 | 129.051 | 1.000 | FULL_READY | 0 | 0 |

## Derived Checks
- Promotion vs no_promotion subsequent TTFT improvement: `66.868%`
- Promotion vs oracle subsequent TTFT delta: `0.285%`
- `PHASE2_INTERNAL_STATE_*` lines: `73`
- `PHASE2_CORE_AUTO_DECISION` lines: `36`
- Core auto request rows: `18`; mismatches: `0`

## Boundary
first-request base/full dual-store is enabled; base is stored to the configured fast target and full to the configured slow target; promotion is approximated by a request-driven full-disk retrieve that writes back to LocalCPU, not by an independent background scheduler
