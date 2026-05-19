# Real KV INT8 Quantization Error Results (2026-05-06)

## Scope / 实验范围

- Model / 模型: `/data1/llm/Qwen/Qwen2.5-7B-Instruct`
- Context length / 上下文长度: 16K (`context_len=16384`)
- Datasets / 数据集: NarrativeQA, HotpotQA, ShareGPT
- Observed KV dtype / 观测到的 KV 类型: `torch.bfloat16`
- Observed chunk shape / 观测到的 chunk 形状: `[2, 28, 256, 512]`
- Run directory / 原始实验目录: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260506_221907_real_kv_int8_error_qwen7b_16k`

## Result Table / 结果表

| Dataset | Chunks | Max abs err | Weighted mean err | Max abs original | Max sampled p99 err | Max sampled p999 err | Chunks > 0.016 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NarrativeQA | 96 | 1.676919 | 0.041102 | 426.000000 | 0.621094 | 1.531250 | 96 |
| HotpotQA | 64 | 1.674643 | 0.040842 | 426.000000 | 0.646284 | 1.549448 | 64 |
| ShareGPT | 64 | 1.666708 | 0.040767 | 424.000000 | 0.626886 | 1.534882 | 64 |
| ALL | 224 | 1.676919 | 0.040932 | 426.000000 | 0.646284 | 1.549448 | 224 |

## Conclusion / 结论

English: The earlier `max_error < 0.016` result was obtained from bounded synthetic BF16 tensors, not from real dataset KV. On real Qwen2.5-7B BF16 KV tensors from NarrativeQA, HotpotQA, and ShareGPT, the maximum absolute error is about `1.677`, and the weighted mean absolute error is about `0.041`.

中文：之前 `max_error < 0.016` 的结果来自有界的合成 BF16 张量，不是真实数据集产生的 KV。现在在 NarrativeQA、HotpotQA、ShareGPT 的真实 Qwen2.5-7B BF16 KV 上验证后，最大绝对误差约为 `1.677`，加权平均绝对误差约为 `0.041`。

## Why It Happens / 原因解释

English: The current INT8 codec uses max-abs based symmetric scaling. Real KV tensors contain large activation outliers, with `max_abs_original` around `426`. Therefore, the worst-case dequantization error is roughly `max_abs / (2 * 127)`, which is about `1.677`. This matches the measured max error.

中文：当前 INT8 codec 使用基于最大绝对值的对称量化。真实 KV 中存在较大的 activation outlier，`max_abs_original` 约为 `426`。因此，最坏情况下的反量化误差大约是 `max_abs / (2 * 127)`，也就是约 `1.677`，这和实测最大误差一致。

## Paper-Safe Statement / 论文中更安全的表述

Use this wording / 建议这样写：

> On bounded synthetic BF16 KV tensors, INT8 round-trip max absolute error is 0.015625. On real Qwen2.5-7B BF16 KV tensors from NarrativeQA, HotpotQA, and ShareGPT, activation outliers increase the max absolute error to about 1.68, while weighted mean absolute error remains about 0.041. End-to-end output match remained 1.0 in the measured real-dataset scenarios, so numeric error and output quality should be reported separately.

## Files / 文件

- Stable JSON / 顶层 JSON: `/home/panzihang/src/LMCache-phase0-codex/REAL_KV_INT8_ERROR_RESULTS_20260506.json`
- Stable CSV / 顶层 CSV: `/home/panzihang/src/LMCache-phase0-codex/REAL_KV_INT8_ERROR_RESULTS_20260506.csv`
- Original run summary / 原始 summary: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260506_221907_real_kv_int8_error_qwen7b_16k/REAL_KV_INT8_ERROR_SUMMARY.json`
- Trace files / 原始 trace: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260506_221907_real_kv_int8_error_qwen7b_16k/*_trace.jsonl`
