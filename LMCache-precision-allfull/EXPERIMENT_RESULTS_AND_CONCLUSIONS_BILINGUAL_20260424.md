# Complete Experiment Results and Conclusions / 完整实验数据与结论（2026-04-24）

## 0. Scope / 范围
This document consolidates the completed LMCache multi-fidelity KV cache experiments: Phase 1 quality-aware base restoration and Phase 2 fidelity promotion. All experiments were run on the remote server under `/home/panzihang/src/LMCache-phase0-codex`; the local workspace was not modified.

本文档整理 LMCache 多质量等级 KV Cache 实验：Phase 1 的 base 表示与调度，以及 Phase 2 的 fidelity promotion。所有实验均在远端服务器执行，本地代码未修改。

Main source artifacts / 主要来源文件：
- `PHASE1_FINAL_RESULTS_SUMMARY_WITH_CACHEGEN_WIRE_BYTES.csv/.json`
- `PHASE1_O2_TAIL_RECOMPUTE_HIGH_SUMMARY.json`
- `PHASE1_LONGBENCH_NARRATIVEQA_16K_MEDIUM_REAL_SUMMARY.csv/.json`
- `PHASE2_RESULTS_SUMMARY.csv/.json`
- `PHASE2_DUAL_WRITE_RESULTS_20260424.md/.csv/.json`
- generated core table / 新生成核心表：`PAPER_CORE_RESULTS_TABLE_20260424.csv`

## 1. Terminology / 术语对照
- `B0 / LMCache full`: original full-fidelity LMCache path. 原始 full KV 路径。
- `Full reference`: full-fidelity reference used for output matching. 用来计算输出一致性的 full 参考运行。
- `O1 / base`: int8 base KV, no layerwise overlap. int8 base 表示，不使用 layerwise overlap。
- `O2 / base+overlap`: int8 base KV with layerwise overlap/prefetch. int8 base 加 layerwise overlap/prefetch。
- `B2 / CacheGen`: CacheGen compression baseline. CacheGen 压缩基线。
- `Promotion`: move a hot prefix from `BASE_READY` to `FULL_READY`. 将热点 prefix 从 base-ready 提升到 full-ready。
- `TTFT`: time to first token. 首 token 延迟。
- `Quality / match rate`: token-level output match against full reference. 相对 full reference 的 token 级输出匹配率。

## 2. Phase 1 Core TTFT Results / Phase 1 核心 TTFT 结果
Numbers are TTFT in milliseconds. `*` marks known negative rows caused by the current vLLM hook/eager-drain limitation; these should be explained as implementation limitations, not as positive results.

数值单位是 ms。带 `*` 的 O2 high-reuse 结果是已知 vLLM hook/eager-drain 限制导致的负结果，论文里需要明确解释。

| Workload 场景 | B0 LMCache full | Full ref | O1 base | O2 base+overlap | B2 CacheGen | O2 vs B0 | Quality 质量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16K high reuse / 16K 高复用 | 399.969 | 89.030 | 245.514 | 539.697 * | 480.647 | -34.9% | 1.0 |
| 16K medium reuse / 16K 中等复用 | 2661.187 | 2306.050 | 2645.990 | 2326.597 | 2605.806 | 12.6% | 1.0 |
| 32K high reuse / 32K 高复用 | 630.572 | 138.480 | 426.631 | 946.760 * | 810.197 | -50.1% | 1.0 |
| 32K medium reuse / 32K 中等复用 | 6660.682 | 5912.124 | 6552.108 | 5977.714 | 6476.894 | 10.3% | 1.0 |

Key interpretation / 关键解释：
- Medium reuse is the clean positive Phase 1 case: O2 improves 16K medium from `2661.187` to `2326.597` ms and 32K medium from `6660.682` to `5977.714` ms.
- High reuse is not where O2 should be claimed as a win in the current implementation: O1 is faster, and O2 pays hook/overlap overhead.
- Output match stayed at `1.0` for O1/O2/B2 in the measured rows.

## 3. Phase 1 Transfer/Storage Size / Phase 1 传输与存储体积
For B0/O1/O2 the size is recorded retrieved MemoryObj size. For CacheGen, `B2 wire GB(est.)` is the serialized wire estimate from the CacheGen serde microbench: serialized/original mean = `0.350483`, compression ratio = `2.853x`.

B0/O1/O2 使用日志记录的 retrieved MemoryObj size；CacheGen 使用序列化后 wire bytes 估计。

| Workload 场景 | B0 GB | O1/O2 base GB | O1/O2 ratio | B2 wire GB(est.) | B2 wire ratio |
| --- | --- | --- | --- | --- | --- |
| 16K high reuse / 16K 高复用 | 0.8750 | 0.4409 | 0.504x | 0.3067 | 0.350x |
| 16K medium reuse / 16K 中等复用 | 0.4375 | 0.2205 | 0.504x | 0.1533 | 0.350x |
| 32K high reuse / 32K 高复用 | 1.7363 | 0.8749 | 0.504x | 0.6085 | 0.350x |
| 32K medium reuse / 32K 中等复用 | 0.8613 | 0.4340 | 0.504x | 0.3019 | 0.350x |

Conclusion / 结论：base roughly halves KV transfer size (`~0.504x` vs B0). CacheGen has a smaller estimated wire footprint (`~0.350x`) but higher TTFT in the main medium-reuse scenarios.

## 4. Phase 1 Retrieve/Store Breakdown / Phase 1 retrieve/store 分解
Each cell is `retrieve_ms/store_ms`; empty store means no store occurred in that hit-only row.

每个单元格格式为 `retrieve_ms/store_ms`；空 store 表示该行没有发生 store。

| Workload 场景 | B0 retrieve/store | O1 retrieve/store | O2 retrieve/store | B2 retrieve/store |
| --- | --- | --- | --- | --- |
| 16K high reuse / 16K 高复用 | 357.440/ | 197.967/ | 460.469/ | 416.946/ |
| 16K medium reuse / 16K 中等复用 | 217.556/188.318 | 111.189/268.229 | 497.782/2187.611 | 197.963/170.120 |
| 32K high reuse / 32K 高复用 | 578.361/ | 366.674/ | 840.543/ | 732.521/ |
| 32K medium reuse / 32K 中等复用 | 422.807/390.582 | 186.074/489.150 | 626.695/5663.891 | 373.612/342.109 |

Important conclusion / 重要结论：O2 medium-reuse wins despite extra base decode/store work because overlap helps the mixed hit/miss workload. O2 high-reuse loses because the current vLLM hook path adds overhead when almost everything is already cache-hit.

## 5. Phase 1 High-Reuse Workaround / Phase 1 high-reuse 负结果修正尝试
A 256-token tail-recompute workaround reduces the O2 high-reuse penalty, but does not make high-reuse O2 the best row.

| Workload 场景 | O2 tail-recompute TTFT | Old O2 delta | B0 delta | O1 delta | Quality | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 16K high reuse / 高复用 | 417.619 | -122.078 | 17.650 | 172.105 | 1.0 | 256-token tail recompute |
| 31K high reuse / 高复用 | 858.512 | -88.248 | 227.940 | 431.881 | 1.0 | 256-token tail recompute |

Interpretation / 解释：tail recompute confirms the negative O2 high-reuse rows are partly implementation-driven, but the paper should still report them honestly as current-limit negative results.

## 6. Real-Token Validation / 真实数据验证
Dataset: LongBench NarrativeQA, 16K medium reuse, controlled prefix overlap.

| Scenario 场景 | TTFT ms | retrieve/store ms | retrieved GB | match 质量 | Note |
| --- | --- | --- | --- | --- | --- |
| b0_full_reference | 2303.576 | 19.826/31.882 | 0.4375 | 1.0 | LongBench NarrativeQA real tokens, 16K medium reuse |
| o1_base_no_prefetch | 2724.949 | 99.820/367.099 | 0.2205 | 1.0 | LongBench NarrativeQA real tokens, 16K medium reuse |
| o2_base_prefetch | 2308.582 | 492.922/2172.111 | 0.2205 | 1.0 | LongBench NarrativeQA real tokens, 16K medium reuse |

Conclusion / 结论：on real NarrativeQA tokens, O2 is near B0 full (`2308.582` vs `2303.576` ms) and faster than O1 (`2724.949` ms), while preserving output match `1.0`.

## 7. Phase 2 Early State-Machine Runs / Phase 2 早期状态机验证
These runs used the original full-materialization proxy before real first-request dual-write existed. They validate the state machine and post-promotion benefit, but are not the final implementation boundary.

这些是双写路径实现前的 proxy 实验，验证状态机和收益趋势，但不是最终双写边界。

| Context 上下文 | No promo subsequent | Promotion subsequent | Oracle | Improvement 收益 | Reuse rate | Quality | Boundary 边界 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 127.789 | 59.282 | 57.706 | 53.6% | 2.0 | 1.0 | proxy, before dual-write |
| 16384 | 448.788 | 111.102 | 120.952 | 75.2% | 2.0 | 1.0 | proxy, before dual-write |

## 8. Phase 2 Real Dual-Write Validation / Phase 2 真实双写验证
### 8.1 Small-model 32K validation / 小模型 32K 验证
Model: `/data1/llm/Llama-3.2-1B`.

| Scenario 场景 | Subseq TTFT ms | Prime TTFT ms | Store total ms | Promotion success | Reuse rate | Quality |
| --- | --- | --- | --- | --- | --- | --- |
| no_promotion | 245.735 | 4882.397 | 871.192 | 0 |  | 1.0 |
| promotion | 101.579 | 4937.201 | 936.390 | 1 | 5.0 | 1.0 |
| oracle | 105.514 | 4237.367 | 264.449 | 0 |  | 1.0 |

Small-model conclusion / 小模型结论：promotion reduced subsequent TTFT from `245.735` to `101.579` ms (`58.662%`) and matched oracle (`105.514` ms) within noise. Reuse rate was `5.0`.

### 8.2 Qwen2.5-7B final Phase 2 result / Qwen2.5-7B 最终 Phase 2 结果
Model: `/data1/llm/Qwen/Qwen2.5-7B-Instruct`. Because Qwen2.5-7B has `max_position_embeddings=32768`, the successful run used `context_len=32760`, `max_model_len=32768`, and `max_tokens=4`.

| Scenario 场景 | State 状态 | First followup TTFT | Mean followup TTFT | Mean subsequent TTFT | Promotion success | Full reuse | Reuse rate | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_promotion | BASE_READY | 562.062 | 544.761 | 541.300 | 0 | 0 |  | 1.0 |
| promotion | FULL_READY | 551.499 | 353.084 | 313.401 | 1 | 5 | 5.0 | 1.0 |
| oracle_full_ready | FULL_READY | 319.280 | 323.821 | 324.729 | 0 | 0 |  | 1.0 |

Qwen result / Qwen 结论：promotion reduced subsequent TTFT from `541.300` to `313.401` ms, saving `227.899` ms per later request (`42.1%`). It is effectively at oracle level (`324.729` ms), with quality `1.0` and reuse rate `5.0`.

## 9. Phase 2 Cost and Break-Even / Phase 2 成本与盈亏平衡
| Model/Context 模型/上下文 | single prime/store | dual prime/store | TTFT overhead | Store overhead | Disk/full copy |
| --- | --- | --- | --- | --- | --- |
| Llama-1B 512 | 48.328 / 37.416 | 63.590 / 52.011 | 15.262 | 14.595 | 17M, 32 files |
| Llama-1B 32768 | 4490.834 / 606.127 | 4882.397 / 871.192 | 391.563 | 265.065 | 1.1G, 128 files |
| Qwen2.5-7B 32760 | 10839.502 / 1003.190 | 11480.835 / 1635.176 | 641.333 | 631.986 | ~1.7363 GiB per prefix, 127 files |

For Qwen2.5-7B 32760-token:
- first-request dual-write TTFT overhead = `641.333` ms;
- each later promoted full-ready request saves `227.899` ms;
- break-even = `2.81` follow-up requests, i.e. positive after about 3 reuses;
- with 5 full-ready reuses, net saving = `5 * 227.899 - 641.333 = 498.162` ms.

中文结论：Qwen 7B 下第一次请求多付约 `641 ms`，但每次后续 full-ready 命中节省约 `228 ms`。因此大约第 3 次后续复用后转正；当前 reuse rate=5.0，整体净收益约 `498 ms`。

## 10. Hot/Cold Negative Test / 冷热混合负向实验
| Prefix 前缀 | final_state | promotion_enqueued | promotion_success | full_reuse_count | reuse_rate |
| --- | --- | --- | --- | --- | --- |
| hot | FULL_READY | 1 | 1 | 4 | 4.0 |
| cold | BASE_READY | 0 | 0 | 0 |  |

Conclusion / 结论：hotness policy correctly avoids promotion-to-fast-tier for cold prefixes. However, the current implementation is eager dual-write: cold prefixes can still write a full copy to slow disk tier on first request. This is storage/I/O overhead and must be stated in the paper.

## 11. What We Can Claim / 可写进论文的结论
1. Base KV reduces transferred/restored size by about `0.50x` while preserving output match `1.0` in the measured workloads.
2. Phase 1 O2 is useful in medium-reuse workloads: 16K medium improves `2661.187 -> 2326.597` ms; 32K medium improves `6660.682 -> 5977.714` ms.
3. O2 high-reuse is currently a negative result due to vLLM hook/eager-drain overhead; this should be reported, not hidden.
4. CacheGen has better estimated wire compression (`~0.35x`) but worse TTFT than O2 in the medium-reuse target rows.
5. Real-token LongBench NarrativeQA validation preserves output match `1.0`; O2 is close to full-reference latency and better than O1.
6. Phase 2 real dual-write is working: first request writes base to fast tier and full to slow tier.
7. Qwen2.5-7B Phase 2 promotion reduces subsequent TTFT by `42.1%` and reaches oracle-level full-ready performance.
8. Break-even is about 3 follow-up requests; at 5 reuses the measured net saving is about `498 ms`.
9. Hot/cold policy avoids wasting fast-tier promotion resources on cold prefixes.

## 12. What We Should Not Overclaim / 不能过度声称的边界
1. Do not claim fully independent background scheduling has been implemented. Current promotion is request-driven full-disk retrieve/writeback.
2. Do not claim layerwise disk retrieve/write-back is complete. Current final Phase 2 promotion result is non-layerwise request-driven promotion.
3. Do not say dual-write is free because it is asynchronous. Disk file writing is async after submit, but first-request D2H/base encode/submit overhead is real and measured.
4. Do not hide storage cost: Qwen2.5-7B 32760-token full slow-tier copy is about `1.7363 GiB` per prefix.
5. Do not present O2 high-reuse rows as a performance win; they are useful as an implementation-limit diagnosis.

## 13. Final Status / 当前阶段状态
Phase 2 experiments can be considered complete if independent background scheduler and layerwise disk retrieve/write-back are treated as future work. The data package now supports paper writing: main TTFT table, transfer-size table, real-token validation, promotion benefit, dual-write cost, break-even, and hot/cold negative test are all recorded.

如果把独立后台调度器和 layerwise disk retrieve/write-back 作为 future work，那么当前 Phase 2 实验可以收尾，进入论文写作和图表整理阶段。

## 14. Source Run Directories / 原始运行目录
- Phase 2 Qwen2.5-7B matrix: `phase2_runs/20260424_163955_32k32760_dualwrite_qwen7b_matrix`
- Phase 2 Qwen2.5-7B single-write baseline: `phase2_runs/20260424_164414_32k32760_singlewrite_qwen7b_baseline`
- Phase 2 hot/cold mixed: `phase2_runs/20260424_150651_mixed_hotcold_512_llama1b`
- Phase 2 Llama-1B 32K no promotion: `phase2_runs/20260424_150933_32k_dualwrite_nopromotion_llama1b`
- Phase 2 Llama-1B 32K promotion: `phase2_runs/20260424_150759_32k_dualwrite_promotion_llama1b`
- Phase 2 Llama-1B 32K oracle: `phase2_runs/20260424_151040_32k_oracle_llama1b`
- Phase 1 source summaries are the CSV/JSON files listed in Section 0.

<!-- REAL_KV_INT8_ERROR_20260506_START -->

## 15. Real KV INT8 Numeric Error / 真实 KV INT8 数值误差

This section corrects the earlier synthetic-only statement about INT8 error.

本节修正之前只基于合成张量得到的 INT8 误差结论。

| Dataset | Chunks | Max abs err | Weighted mean err | Max abs original | Max sampled p99 err | Max sampled p999 err | Chunks > 0.016 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NarrativeQA | 96 | 1.676919 | 0.041102 | 426.000000 | 0.621094 | 1.531250 | 96 |
| HotpotQA | 64 | 1.674643 | 0.040842 | 426.000000 | 0.646284 | 1.549448 | 64 |
| ShareGPT | 64 | 1.666708 | 0.040767 | 424.000000 | 0.626886 | 1.534882 | 64 |
| ALL | 224 | 1.676919 | 0.040932 | 426.000000 | 0.646284 | 1.549448 | 224 |

Conclusion / 结论：

- Synthetic result / 合成数据结果: bounded BF16 tensors had max error `0.015625`.
- Real KV result / 真实 KV 结果: real Qwen2.5-7B BF16 KV had max error about `1.677` and weighted mean error about `0.041`.
- Cause / 原因: real KV contains large activation outliers (`max_abs_original` about `426`), so max-abs INT8 scaling naturally gives a worst-case error near `1.677`.
- Paper handling / 论文处理: report numeric error and end-to-end output quality separately. Do not claim real KV max error is below `0.016`.

Stable files / 稳定文件：

- `/home/panzihang/src/LMCache-phase0-codex/REAL_KV_INT8_ERROR_RESULTS_20260506.md`
- `/home/panzihang/src/LMCache-phase0-codex/REAL_KV_INT8_ERROR_RESULTS_20260506.json`
- `/home/panzihang/src/LMCache-phase0-codex/REAL_KV_INT8_ERROR_RESULTS_20260506.csv`

<!-- REAL_KV_INT8_ERROR_20260506_END -->

<!-- LONG_OUTPUT_MATCH_20260507_START -->

## 16. Long-output Output Match / 长输出输出匹配验证

This section was added because the real-KV INT8 numeric error is tail-heavy: mean absolute error is small, but max error is large.

本节用于回应真实 KV INT8 误差的 tail-heavy 特征：平均绝对误差较小，但最大误差较大。

| Model | Dataset | Context | Max tokens | Ignore EOS | Actual output tokens | Base prime match | Base followup match | Followup exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | narrativeqa | 16384 | 64 | False | 64 | 1.000000 | 0.062500 | False |
| Llama-3.2-1B | hotpotqa | 8192 | 64 | False | 64 | 1.000000 | 1.000000 | True |

Conclusion / 结论：

- `max_tokens=4` came from the previous Qwen2.5-7B run script and PHASE config logs, not from inference.
- Low mean KV error alone is not enough to prove long-output safety.
- Auxiliary Llama-3.2-1B results are mixed: NarrativeQA 16K diverged after INT8 base-cache retrieval (`0.0625` match over 64 tokens), while HotpotQA 8K still matched exactly.
- Qwen2.5-7B 16K/64-token NarrativeQA and HotpotQA must still be run when GPU memory is available.

Files / 文件：

- `/home/panzihang/src/LMCache-phase0-codex/LONG_OUTPUT_MATCH_RESULTS_20260507.md`
- `/home/panzihang/src/LMCache-phase0-codex/LONG_OUTPUT_MATCH_RESULTS_20260507.json`
- `/home/panzihang/src/LMCache-phase0-codex/LONG_OUTPUT_MATCH_RESULTS_20260507.csv`

<!-- LONG_OUTPUT_MATCH_20260507_END -->
