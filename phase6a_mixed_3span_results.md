# Phase 6A Mixed 3-span Routing Results
## Scope
- Policy: `mixed-3span` = first chunk full, middle chunks base, last 2 chunks full.
- Model benchmark: `/data1/llm/Qwen/Qwen2-1.5B`, synthetic `long-doc-qa`, document length 2048.
- Chunk count: 8 chunks, `full_chunks=3`, `base_chunks=5`.
- Qwen2 bytes ratio: `0.692383`.

## Slot Order Validation
- Run: `/home/panzihang/src/experience/runs/phase6a_20260518_200913_mixed_slot_validation_gpu2`
- Hit chunks: `8/8`

| chunk | tier | cosine | max_abs_error |
|---:|---|---:|---:|
| 0 | full | 1.000000000 | 0.000000 |
| 1 | base | 0.999981344 | 0.023438 |
| 2 | base | 0.999981403 | 0.023438 |
| 3 | base | 0.999981463 | 0.031250 |
| 4 | base | 0.999981642 | 0.023438 |
| 5 | base | 0.999981403 | 0.015625 |
| 6 | full | 1.000000000 | 0.000000 |
| 7 | full | 1.000000000 | 0.000000 |

## Benchmarks

| run | success | failed | mean TTFT ms | elapsed s | output tok/s | store events | retrieve events |
|---|---:|---:|---:|---:|---:|---:|---:|
| qpd=1 official-shape | 3 | 0 | 34.404 | 1.987 | 193.255 | 3 | 0 |
| forced retrieve qpd=2 | 6 | 0 | 42.899 | 3.969 | 193.499 | 3 | 6 |

## Mixed Logs
- qpd=1 store: `[('8', '3', '5', '0.692383', '75.3819'), ('8', '3', '5', '0.692383', '93.5907'), ('8', '3', '5', '0.692383', '87.2569')]`
- forced retrieve store: `[('8', '3', '5', '0.692383', '75.1282'), ('8', '3', '5', '0.692383', '94.9405'), ('8', '3', '5', '0.692383', '89.8770')]`
- forced retrieve retrieve: `[('8', '3', '5', '0.692383', '1.0302'), ('8', '3', '5', '0.692383', '0.5543'), ('8', '3', '5', '0.692383', '0.5296'), ('8', '3', '5', '0.692383', '0.4825'), ('8', '3', '5', '0.692383', '0.5040'), ('8', '3', '5', '0.692383', '0.4988')]`

## Artifacts
- qpd=1: `/home/panzihang/src/experience/runs/phase6a_20260518_201534_mixed_3span_qpd1_gpu3`
- forced retrieve: `/home/panzihang/src/experience/runs/phase6a_20260518_201347_mixed_3span_forced_retrieve_gpu3`
