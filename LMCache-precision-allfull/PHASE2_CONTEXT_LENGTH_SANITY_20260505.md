# Phase2 Context-Length Sanity Check (2026-05-05)

## Question
We repeatedly observed about 66-67% TTFT reduction from promotion. This check tests whether that percentage is fixed by an experiment bug or varies with context length/workload.

## Valid Runs
| workload | context tokens | no_promotion subsequent TTFT ms | promotion subsequent TTFT ms | oracle subsequent TTFT ms | improvement | promo-oracle delta | core FULL_READY lines | core PROMOTING lines |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa_8k | 8192 | 183.182 | 76.936 | 76.866 | 58.000% | 0.091% | 22 | 0 |
| hotpotqa_16k | 16384 | 366.683 | 124.491 | 127.265 | 66.049% | -2.180% | 22 | 0 |
| sharegpt_16k | 16384 | 388.808 | 128.820 | 129.188 | 66.868% | -0.285% | 22 | 0 |
| sharegpt_32k_aligned | 32512 | 798.248 | 249.027 | 249.290 | 68.803% | -0.105% | 22 | 0 |

## HotpotQA 32K Feasibility
- Qwen-tokenized HotpotQA prompts with the current construction: `>=8192`: 173 rows, `>=16384`: 65 rows, `>=32768`: 0 rows.
- Longest HotpotQA prompt found: 17625 tokens.
- Therefore a real HotpotQA 32K Phase2 run is not possible without artificial padding/repetition, which would no longer be a clean HotpotQA real-data run.

## Excluded 32K Attempt
- Excluded `phase2_runs/20260505_real_sharegpt_32k32752_qwen7b_internal_contextcheck`: context_len=32752 is not divisible by LMCache chunk_size=256; LMCache retrieved only 32512 full-chunk tokens and internal state for the full prefix stayed PROMOTING.
- Evidence: core FULL_READY lines `0`, core PROMOTING lines `52`, repeated `Retrieved 32512 out of 32512` lines `20`.

## Conclusion
- The improvement is not fixed at 66-67%: HotpotQA 8K gives about 58%, while 16K gives about 66%.
- ShareGPT 32K chunk-aligned gives about 69%, with promotion matching oracle full closely.
- The 16K repeated 66-67% is plausible because the no_promotion base decode/retrieve path and full fast-hit path have a fairly stable ratio at that size, not because all measurements are hard-coded or broken.
- For future 32K LMCache runs, use chunk-aligned prompt lengths such as 32512 when max model length is 32768 and generation needs extra tokens.
