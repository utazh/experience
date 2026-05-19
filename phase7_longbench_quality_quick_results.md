# Phase 7 LongBench Quick Validation (Qwen2-7B-Instruct, 30+30)
有效结果使用 LMCache L1=32GB。此前 L1=4GB 的尝试只有 4 个 RETRIEVE 事件，大部分样本未真正使用外部 KV retrieve，因此不作为质量结论。
数据集：qasper 前 30 条、multifieldqa_en 前 30 条；prompt 来自 LongBench 官方 dataset2prompt；指标为 LongBench qa_f1_score。每条样本先 warm 一次写入 LMCache，第二次相同 prompt streaming 生成并计分。
| policy | retrieve bytes_ratio | mean TTFT (ms) | mean latency (ms) | overall F1 | qasper F1 | multifieldqa_en F1 | retrieve chunks full/base | pred diff vs all-full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-full | 1.000 | 125.72 | 524.42 | 45.34 | 46.08 | 44.60 | 0/0 | 0 |
| threshold=0.50 | 0.784 | 128.56 | 516.70 | 44.41 | 44.42 | 44.39 | 789/619 | 10 |
| threshold=0.75 | 0.660 | 124.37 | 526.77 | 44.93 | 46.37 | 43.49 | 436/972 | 10 |
| all-base | 0.508 | 123.32 | 504.33 | 41.71 | 41.35 | 42.07 | 0/1408 | 15 |

## Notes
- all-full 没有 MP_MIXED_RETRIEVE 字节日志，bytes_ratio 按 1.0 记录。
- threshold=0.50 retrieve weighted bytes_ratio=0.784；threshold=0.75 为 0.660；all-base 为 0.508。
- L1=32GB 后第二轮请求 TTFT 降到约 124-129ms，说明 external KV retrieve 路径有效；L1=4GB 版本因缓存逐出导致 TTFT 约 1.35s，不用于质量结论。
- qasper 有 4/30 条 prompt 被截断，multifieldqa_en 有 17/30 条被截断；max_model_len=8192。
- 预测文本差异较少：threshold=0.50 相比 all-full 有 10 条不同，threshold=0.75 有 10 条不同，all-base 有 15 条不同。

## Run Directories
- all-full: `/home/panzihang/src/experience/runs/quality_quick_20260518_231106_all-full_qwen2_7b_gpu3`
- threshold=0.50: `/home/panzihang/src/experience/runs/quality_quick_20260518_231345_threshold-mixed_0.50_qwen2_7b_gpu3`
- threshold=0.75: `/home/panzihang/src/experience/runs/quality_quick_20260518_231623_threshold-mixed_0.75_qwen2_7b_gpu3`
- all-base: `/home/panzihang/src/experience/runs/quality_quick_20260518_231905_all-base_qwen2_7b_gpu3`
