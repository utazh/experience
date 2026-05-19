# Phase2 Real HotpotQA 16K Internal-State 2026-05-04

- Run dir: `/home/panzihang/src/LMCache-phase0-codex/phase2_runs/20260504_real_hotpotqa_16k_qwen7b_internal`
- Model: `/data1/llm/Qwen/Qwen2.5-7B-Instruct`
- Dataset: `/data1/datasets/LongBench/data/hotpotqa.jsonl`
- State source: LMCache internal fidelity state store
- Promotion mode: request-driven full disk retrieve/writeback

| scenario | mean_subsequent_ttft_ms | mean_followup_ttft_ms | match_rate | final_state | full_reuse_count | promotion_success |
|---|---:|---:|---:|---|---:|---:|
| no_promotion | 366.683 | 365.806 | 1.0 | BASE_READY | 0 | 0 |
| promotion | 124.491 | 166.565 | 1.0 | FULL_READY | 5 | 1 |
| oracle_full_ready | 127.265 | 128.335 | 1.0 | FULL_READY | 0 | 0 |

## Derived

- Promotion vs no_promotion subsequent TTFT improvement: 66.049%
- Promotion vs oracle subsequent TTFT delta: -2.180%
- PHASE2_INTERNAL_STATE_* lines: 73
- PHASE2_CORE_AUTO_DECISION lines: 36

## Dataset Row

- row_index: 1
- record_id: `d542fcd45bbf5112eee9127f04ae060a887c8ef2050aa160`
- question: The actor that plays Phileas Fogg in "Around the World in 80 Days", co-starred with Gary Cooper in a 1939 Goldwyn Productions film based on a novel by what author?
- answers: ['Charles L. Clifford']
