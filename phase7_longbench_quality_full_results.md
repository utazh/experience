# Phase 7 LongBench Full Quality Results
Qwen2-7B-Instruct；qasper 200 条 + multifieldqa_en 150 条。all-full 使用单次 350 样本运行；threshold=0.75 和 all-base 使用 50 条分片运行后聚合，避免长进程 mixed/base codec 累积触发 CUDA invalid argument。指标使用 LongBench qa_f1_score。
| policy | retrieve bytes_ratio | mean TTFT (ms) | mean latency (ms) | overall F1 | qasper F1 | multifieldqa_en F1 | retrieve full/base chunks | pred diff vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-full | 1.000 | 133.48 | 546.63 | 46.99 | 46.18 | 48.07 | 0/0 | 0 |
| threshold=0.75 | 0.665 | 125.34 | 528.23 | 46.24 | 45.28 | 47.51 | 2432/5170 | 91 |
| all-base | 0.508 | 125.13 | 530.04 | 45.75 | 44.92 | 46.86 | 0/7615 | 108 |

## Notes
- all-full run: `/home/panzihang/src/experience/runs/quality_quick_20260519_094719_all-full_qwen2_7b_gpu3`
- shard root: `/home/panzihang/src/experience/runs/quality_full_shards_20260519_103503_qwen2_7b_l1_32gb`
- qasper truncation count: all-full/threshold/all-base are shown in CSV; max_model_len=8192.
- A monolithic threshold=0.75 run failed around qasper sample 157 with LMCache CUDA invalid argument; the same sample succeeds alone, so final mixed/base results use 50-sample shards.
- `retrieve_bytes_ratio` is computed from `MP_MIXED_RETRIEVE` only; STORE events are excluded from transfer-ratio reporting.
- Logs still contain a few nonfatal LMCache ERROR lines for non chunk-aligned key ranges / shutdown descriptors; every listed run returned rc=0 and produced summaries, but this should be tracked as an implementation cleanup item.
