# Phase 6B Threshold Mixed Results

## Scope
- Policy: `threshold-mixed`
- Score: `max(sink_score, recent_score)`
- `recent_score = (chunk_index + 1) / total_chunks`
- `score >= threshold -> full`, otherwise base
- Threshold tested in benchmark: `0.75`
- For 8 chunks: expected `full_chunks=4`, `base_chunks=4`

Note: because `score >= threshold` selects full, raising threshold makes full chunk count monotonic non-increasing. Unit tests assert this direction.

## Unit Tests
- Threshold sensitivity: threshold `0.25/0.50/0.75` gives full counts `8/6/4`.
- Sink protection: threshold `0.9` keeps chunk 0 full.
- Fallback: invalid/non-standard span shapes degrade to all-full.

## Benchmark Results

| run | success | failed | mean TTFT ms | elapsed s | output tok/s | full/base chunks | bytes ratio |
|---|---:|---:|---:|---:|---:|---|---:|
| threshold=0.75 qpd=1 | 3 | 0 | 33.985 | 1.986 | 193.318 | 4/4 | 0.753906 |
| threshold=0.75 forced retrieve | 6 | 0 | 45.925 | 3.983 | 192.805 | 4/4 | 0.753906 |
| Phase 6A 3-span forced retrieve | 6 | 0 | 42.899 | 3.969 | 193.499 | 3/5 | 0.692383 |

## Log Evidence
- qpd=1 store events: `[('8', '4', '4', '0.753906', '76.5085'), ('8', '4', '4', '0.753906', '85.5245'), ('8', '4', '4', '0.753906', '85.9974')]`
- forced retrieve store events: `[('8', '4', '4', '0.753906', '74.2625'), ('8', '4', '4', '0.753906', '83.0911'), ('8', '4', '4', '0.753906', '81.5994')]`
- forced retrieve retrieve events: `[('8', '4', '4', '0.753906', '0.8422'), ('8', '4', '4', '0.753906', '0.4971'), ('8', '4', '4', '0.753906', '0.4978'), ('8', '4', '4', '0.753906', '0.4925'), ('8', '4', '4', '0.753906', '0.4897'), ('8', '4', '4', '0.753906', '0.4956')]`

## Artifacts
- qpd=1: `/home/panzihang/src/experience/runs/phase6b_20260518_212024_threshold_0.75_qpd1_gpu3`
- forced retrieve: `/home/panzihang/src/experience/runs/phase6b_20260518_211842_threshold_0.75_forced_retrieve_gpu3`
