# Phase 6B Threshold Sweep Results

Model/workload: Qwen2-1.5B, synthetic `long-doc-qa`, qpd=1, document length 2048.

| policy | threshold | bytes_ratio | mean TTFT ms | output tok/s | full_chunks | base_chunks |
|---|---:|---:|---:|---:|---:|---:|
| precision all-full | - | 1.000000 | 34.493 | 193.040 | 8 | 0 |
| threshold mixed | 0.25 | 1.000000 | 36.201 | 192.540 | 8 | 0 |
| threshold mixed | 0.50 | 0.876953 | 33.520 | 193.106 | 6 | 2 |
| threshold mixed | 0.75 | 0.753906 | 33.985 | 193.318 | 4 | 4 |
| all-base gpu codec | - | 0.507812 | 33.868 | 193.071 | 0 | 8 |

## Notes
- `threshold=0.25` selects all chunks as full because chunk 1 has recent score 0.25 and the rule is `score >= threshold -> full`.
- The bytes ratios match the expected Qwen2-1.5B full/base mixture.
- TTFT differences at this smoke scale include measurement noise; chunk counts and byte ratios are the primary sweep validation.

## Artifacts
- precision all-full -: `/home/panzihang/src/experience/runs/phase1_20260518_134027_precision_allfull/bench/bench_summary.json`
- threshold mixed 0.25: `/home/panzihang/src/experience/runs/phase6b_20260518_213504_threshold_0.25_qpd1_gpu3_valid`
- threshold mixed 0.50: `/home/panzihang/src/experience/runs/phase6b_20260518_213610_threshold_0.50_qpd1_gpu3_valid`
- threshold mixed 0.75: `/home/panzihang/src/experience/runs/phase6b_20260518_212024_threshold_0.75_qpd1_gpu3`
- all-base gpu codec -: `/home/panzihang/src/experience/runs/phase5_gpu_codec_20260518_192802_mp_allbase_int8_gpu3`
