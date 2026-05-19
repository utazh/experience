# NarrativeQA F1/EM Cached Fidelity Quality (2026-05-04)

## Setup
- Model: `/data1/llm/Qwen/Qwen2.5-7B-Instruct`
- Dataset: `/data1/datasets/LongBench/data/narrativeqa.jsonl`
- Context length: 16384 tokens
- Samples: 20 LongBench NarrativeQA rows, start_index=0
- Max generation: 32 tokens, greedy decoding, stop at newline
- Metric: normalized token F1 and exact match, max over gold answers
- Compared modes: `reference_uncached`, `base_cached`, `full_cached`

## Results
| mode | samples | mean F1 | mean EM | median F1 | mean TTFT ms | mean E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| reference_uncached | 20 | 0.192400 | 0.000000 | 0.144516 | 2355.033650 | 2895.127050 |
| base_cached | 20 | 0.206171 | 0.000000 | 0.180000 | 886.874000 | 1423.263650 |
| full_cached | 20 | 0.191689 | 0.000000 | 0.129032 | 128.904050 | 666.028400 |

## Comparison
- Base cached minus full cached mean F1: `0.014482`
- Base cached minus reference uncached mean F1: `0.013771`
- Base/full EM difference: `0.000000`
- Per-sample F1: base > full on `5` samples, base < full on `7` samples, equal on `8` samples.
- Raw prediction equality: base == full on `4/20` samples.

## Boundary
Scores are computed on cached follow-up generations. reference_uncached uses full fidelity with skip_save; base_cached primes then retrieves INT8 base KV; full_cached primes then retrieves full KV. Base cleanup is disabled for this A/B quality run so base and full variants can coexist for the same prompts.

## Notes
- This is a 20-sample pilot quality run, not a full LongBench leaderboard result.
- EM is zero for all modes because NarrativeQA answers are abstractive and exact string matching is strict; F1 is the more informative metric here.
- The result supports that cached INT8 base does not show a measurable F1 drop versus cached full on this pilot set; the mean F1 is slightly higher for base, which should be reported as no observed degradation rather than an improvement claim.
