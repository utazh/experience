# Scheduler Overlap Opportunity Diagnostic (2026-05-05)

## Setup
- No scheduler or LMCache core behavior was changed.
- Script: `benchmarks/quality_aware_kv/phase2/diagnose_scheduler_overlap.py`
- Method: prime one ShareGPT prefix into full local-disk cache, run one MISS baseline, then submit `pair_full_hit` followed by `pair_miss` after a small delay.
- The full-hit request uses non-layerwise LMCache retrieve; the MISS request uses `lmcache.skip_save=True` to avoid store overhead.

## Results
| run | context | miss baseline TTFT ms | pair full-hit TTFT ms | pair MISS TTFT ms | MISS excess ms | full retrieve cost ms | MISS relation to retrieve | retrieve-overlap upper bound ms | full-hit-wait upper bound ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| smoke_2k_llama1b | 2048 | 98.534 | 30.570 | 674.703 | 576.169 | 9.911 | during_retrieve | 6.000 | 18.000 |
| formal_16k_qwen7b | 16384 | 4083.168 | 343.041 | 4431.189 | 348.021 | 100.469 | before_retrieve_start | 100.469 | 321.000 |

## Formal 16K Interpretation
- MISS submit: `2026-05-05 18:25:17.555000`.
- Full retrieve window: `2026-05-05 18:25:17.591531` to `2026-05-05 18:25:17.692000`.
- MISS relation to retrieve: `before_retrieve_start`; queued before/during retrieve: `True`.
- Logged LMCache full retrieve cost: `100.469` ms.
- Full-hit pre-token wait after MISS submit: `321.000` ms.
- MISS baseline TTFT: `4083.168` ms; pair MISS TTFT: `4431.189` ms.
- Pair MISS excess over baseline: `348.021` ms.

## Conclusion
- There is real queue-level overlap opportunity: the MISS request was already queued while the full-hit request was still before first token and while LMCache retrieve was running.
- In the 16K Qwen run, the direct LMCache retrieve component is about 100 ms, while the full-hit request occupies the scheduler/model step for about 321 ms after the MISS is submitted. The observed MISS excess is about 348 ms.
- Therefore an async-load connector could plausibly save hundreds of milliseconds in this collision pattern, mainly by allowing MISS prefill to start while the full-hit request is in its load/pre-token phase. It will not remove the full 4s MISS prefill compute time.
