# Long-output Output Match Validation (2026-05-07)

## Why this was added / 为什么补这个实验

The real-KV numeric experiment showed a tail-heavy error distribution: max error is large, but mean absolute error is small. That does not by itself prove long autoregressive decoding is safe, because small KV perturbations can be amplified after several generated tokens.

真实 KV 数值实验显示误差是 tail-heavy：最大误差较大，但平均绝对误差较小。这不能直接证明长输出安全，因为自回归生成过程中，小的 KV 扰动可能在后续 token 中被放大。

## How Claude knew `max_tokens=4` / Claude 怎么看出 `max_tokens=4`

- Previous run script: `phase1_runs/20260506_221907_real_kv_int8_error_qwen7b_16k/run.sh` contains `--max-tokens 4` for NarrativeQA, HotpotQA, and ShareGPT.
- Previous logs: `narrativeqa.log`, `hotpotqa.log`, and `sharegpt.log` contain config rows with `"max_tokens": 4`.

## Current GPU limitation / 当前 GPU 限制

Qwen2.5-7B could not be rerun now: GPU0 has only about 8.4 GiB free, GPUs 1/2 are occupied by another user's vLLM worker, and GPU3 is running another user's training job. Therefore the following is an auxiliary Llama-3.2-1B validation, not a replacement for Qwen2.5-7B.

当前无法重跑 Qwen2.5-7B：GPU0 只有约 8.4 GiB 可用，GPU1/2 被其他用户的 vLLM worker 占满，GPU3 有其他用户训练任务。因此下面是 Llama-3.2-1B 的辅助验证，不能替代 Qwen2.5-7B 主结果。

## Results / 结果

| Model | Dataset | Context | Max tokens | Ignore EOS | Actual output tokens | Base prime match | Base followup match | Followup exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | narrativeqa | 16384 | 64 | False | 64 | 1.000000 | 0.062500 | False |
| Llama-3.2-1B | hotpotqa | 8192 | 64 | False | 64 | 1.000000 | 1.000000 | True |

## Interpretation / 解释

English: The auxiliary long-output check is mixed. On NarrativeQA 16K, the base prime request still matches the full reference exactly, but the base followup request that retrieves from INT8 base cache only matches 4/64 tokens (`0.0625`). On HotpotQA 8K, the base followup still matches exactly for 64 generated tokens. This means low mean absolute KV error is not sufficient evidence by itself; output quality must be validated at the target output length.

中文：辅助长输出验证结果是混合的。NarrativeQA 16K 下，base prime 请求仍然和 full reference 完全一致，但真正从 INT8 base cache 取回的 base followup 只匹配 4/64 个 token（`0.0625`）。HotpotQA 8K 下，64 token followup 仍然完全匹配。这说明较低 mean absolute KV error 本身证据不足，必须在目标输出长度上验证输出质量。

## Paper handling / 论文处理建议

Do not claim that real-KV INT8 is long-output safe based only on `max_tokens=4` output match or low mean error. A safe statement is:

> Real KV quantization error is tail-heavy: mean absolute error remains low, but long-output exact match can be dataset/model dependent. We therefore report numeric error separately from end-to-end output quality and validate output match at the target generation length.

Remaining required experiment / 仍需补的主实验：Qwen2.5-7B, NarrativeQA/HotpotQA, 16K, `max_tokens=64`, once enough GPU memory is available.

## Files / 文件

- JSON: `/home/panzihang/src/LMCache-phase0-codex/LONG_OUTPUT_MATCH_RESULTS_20260507.json`
- CSV: `/home/panzihang/src/LMCache-phase0-codex/LONG_OUTPUT_MATCH_RESULTS_20260507.csv`
- NarrativeQA 16K no-ignore run: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260507_1740_long_output_match_llama1b_narrativeqa_16k_64_noignore`
- HotpotQA 8K no-ignore run: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260507_1740_long_output_match_llama1b_hotpotqa_8k_64_noignore`
- Auxiliary ignore-EOS run: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260507_1731_long_output_match_llama1b_16k_64`
