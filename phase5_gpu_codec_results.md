
# Phase 5 GPU-side INT8 Base Codec Results

## Scope
- Code copy: `/home/panzihang/src/experience/LMCache-precision-allfull`
- Model: `/data1/llm/Qwen/Qwen2-1.5B`
- Workload: official synthetic `long-doc-qa`, document length 2048, KV volume 0.2 GB
- Policy: `LMCACHE_PRECISION_POLICY=all-base`, `LMCACHE_BASE_CODEC=int8`
- Base layout: INT8 KV tensor + FP32 per-vector scale, bytes ratio `0.507812`

## Validation
- Unit tests: `15 passed, 2 warnings`
- Real model KV quality validation: layers `[0, 14, 27]`, cosine threshold `0.999`, all pass `True`

| layer | cosine_similarity | max_abs_error | mean_abs_error |
|---:|---:|---:|---:|
| 0 | 0.999967337 | 2.000000 | 0.315349 |
| 14 | 0.999927640 | 0.062500 | 0.012105 |
| 27 | 0.999899089 | 0.093750 | 0.020389 |


## Benchmark Results

| run | successful | failed | mean TTFT ms | elapsed s | output tok/s | store events | retrieve events |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPU codec qpd=1 | 3 | 0 | 33.868 | 1.989 | 193.071 | 3 | 0 |
| GPU codec forced retrieve qpd=2 | 6 | 0 | 40.006 | 3.970 | 193.428 | 3 | 6 |

## Codec Timing Logs

- qpd=1 store encode_ms: ['76.4829', '91.0774', '89.5634']
- forced retrieve store encode_ms: ['66.6497', '89.7124', '90.8935']
- forced retrieve decode_enqueue_ms: ['1.1688', '0.7962', '0.7934', '0.7129', '0.8534', '0.8622']

## Artifacts
- qpd=1 run: `/home/panzihang/src/experience/runs/phase5_gpu_codec_20260518_192802_mp_allbase_int8_gpu3`
- forced retrieve run: `/home/panzihang/src/experience/runs/phase5_gpu_codec_20260518_192941_mp_int8_retrieve_forced_gpu3`
- quality validation: `/home/panzihang/src/experience/runs/phase5_gpu_codec_quality_20260518_192314/quality_summary.json`
