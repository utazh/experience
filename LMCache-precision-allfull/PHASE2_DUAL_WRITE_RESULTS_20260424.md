# Phase 2 Dual-Write Results (2026-04-24)

## Scope
All work was done on the remote server under `/home/panzihang/src/LMCache-phase0-codex`. The local workspace was not modified.

Implemented and tested:
- first-request dual-store path for fidelity base requests: base variant to `LocalCPUBackend`, full variant to `LocalDiskBackend`;
- request-driven full-disk retrieve/writeback promotion probe for non-layerwise runs;
- hot/cold mixed negative test;
- 32K-class hot-prefix runs on both `/data1/llm/Llama-3.2-1B` and `/data1/llm/Qwen/Qwen2.5-7B-Instruct`.

For Qwen2.5-7B, `context_len=32760` and `max_model_len=32768` were used because the model advertises `max_position_embeddings=32768` and the request generates `max_tokens=4`. The direct `32768 + 4 + margin` attempt is recorded as a failed max-length configuration, not as a data-path failure.

## Code Changes
- `lmcache/v1/cache_engine.py`: dual-store helpers and base-request base/full writes.
- `benchmarks/quality_aware_kv/phase2/run_phase2_hot_prefix.py`: dual-write CLI/config logging and promotion boundary labels.
- `benchmarks/quality_aware_kv/phase2/run_phase2_hot_cold_mixed.py`: new hot/cold mixed negative experiment.

## Qwen2.5-7B 32K Main Result
| scenario | final_state | first_followup_ttft_ms | mean_followup_ttft_ms | mean_subsequent_ttft_ms | promotion_success | full_reuse_count | reuse_rate | match_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_promotion | BASE_READY | 562.062 | 544.761 | 541.300 | 0 | 0 |  | 1.0 |
| promotion | FULL_READY | 551.499 | 353.084 | 313.401 | 1 | 5 | 5.0 | 1.0 |
| oracle_full_ready | FULL_READY | 319.280 | 323.821 | 324.729 | 0 | 0 |  | 1.0 |

Promotion reduced subsequent TTFT by `227.899` ms (`42.102`%) versus no promotion. The promoted path was within noise of oracle full-ready: `313.401` ms versus `324.729` ms (`-11.328` ms delta, ratio `0.965`). Reuse rate was `5.0`: one successful promotion served five later full-ready hits.

Run: `phase2_runs/20260424_163955_32k32760_dualwrite_qwen7b_matrix`

## First-Request Store Cost
| context | model | mode | prime TTFT ms | store log total ms | notes |
| ---: | --- | --- | ---: | ---: | --- |
| 512 | Llama-3.2-1B | single-write base | 48.328 | 37.416 | baseline |
| 512 | Llama-3.2-1B | dual-write base+full-disk | 63.590 | 52.011 | +15.262 ms prime TTFT, +14.595 ms store |
| 32768 | Llama-3.2-1B | single-write base | 4490.834 | 606.127 | 32K baseline |
| 32768 | Llama-3.2-1B | dual-write base+full-disk | 4882.397 | 871.192 | +391.563 ms prime TTFT, +265.065 ms store |
| 32760 | Qwen2.5-7B | single-write base | 10839.502 | 1003.190 | 7B baseline |
| 32760 | Qwen2.5-7B | dual-write base+full-disk | 11480.835 | 1635.176 | +641.333 ms prime TTFT (5.917%), +631.986 ms store |

For the Qwen2.5-7B dual-write no-promotion run, logged base-encode time was `1418.462` ms; full-submit-to-disk was only `8.972` ms. This means LocalDisk file writing is submitted asynchronously, but the first-request path still pays D2H capture and base encoding. The measured TTFT/store overhead should be reported as real first-request overhead, not hidden background cost.

## Slow-Tier Storage Footprint
| context | model | run | disk usage | files | per-prefix full footprint |
| ---: | --- | --- | ---: | ---: | ---: |
| 512 | Llama-3.2-1B | `phase2_runs/20260424_145840_dualwrite_smoke_512_llama1b` | 17M | 32 | about 17M |
| 32768 | Llama-3.2-1B | `phase2_runs/20260424_150759_32k_dualwrite_promotion_llama1b` | 1.1G | 128 | about 1.1G |
| 32760 | Qwen2.5-7B | `phase2_runs/20260424_163955_32k32760_dualwrite_qwen7b_matrix` | 5593157632 bytes total | 381 | 1864385877 bytes, 127 files, about 1.7363 GiB from LMCache logs |

The Qwen matrix contains three full prefixes (`no_promotion`, `promotion`, and `oracle_full_ready`), so total disk usage is not a single-prefix number.

## Eager Dual-Write Boundary
The current implementation is eager: a first request for a base prefix writes the base variant to the fast tier and the full variant to the slow tier before we know whether that prefix will become hot. The hotness policy controls promotion into the fast tier; it does not suppress slow-tier full writes for cold prefixes.

This is a valid first implementation, but the paper should describe the cold-prefix full write as storage/I/O overhead. A future lazy variant could write the full slow-tier copy only after a prefix crosses the hotness threshold, trading lower cold-prefix write amplification for less promotion readiness.

## 32K Hot Prefix (Llama-3.2-1B Validation)
| scenario | final_state | mean_subsequent_ttft_ms | promotion_success | full_reuse_count | reuse_rate | match_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no_promotion | BASE_READY | 245.735 | 0 | 0 |  | 1.0 |
| promotion | FULL_READY | 101.579 | 1 | 5 | 5.0 | 1.0 |
| oracle_full_ready | FULL_READY | 105.514 | 0 | 0 |  | 1.0 |

These small-model runs validated the dual-write/full-disk retrieve-writeback path before the 7B run became available.

## Hot/Cold Mixed Negative Test
Run: `phase2_runs/20260424_150651_mixed_hotcold_512_llama1b`

| prefix | final_state | access_count | promotion_enqueued | promotion_success | full_reuse_count | reuse_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hot | FULL_READY | 6 | 1 | 1 | 4 | 4.0 |
| cold | BASE_READY | 1 | 0 | 0 | 0 |  |

This validates that cold prefixes do not consume promotion-to-fast-tier resources. With eager dual-write, cold prefixes can still consume slow-tier storage.

## Run Directories
- Qwen2.5-7B 32760-token dual-write matrix: `phase2_runs/20260424_163955_32k32760_dualwrite_qwen7b_matrix`
- Qwen2.5-7B 32760-token single-write baseline: `phase2_runs/20260424_164414_32k32760_singlewrite_qwen7b_baseline`
- Qwen2.5-7B failed 32768-token attempt: `phase2_runs/20260424_163903_32k_dualwrite_qwen7b_matrix`
- 512 single-write baseline: `phase2_runs/20260424_150033_singlewrite_baseline_512_llama1b`
- 512 layerwise dual-write smoke: `phase2_runs/20260424_145840_dualwrite_smoke_512_llama1b`
- 512 non-layerwise disk promotion/reuse: `phase2_runs/20260424_150420_dualwrite_diskpromotion_512_llama1b`
- 512 hot/cold mixed: `phase2_runs/20260424_150651_mixed_hotcold_512_llama1b`
- 32K single-write baseline: `phase2_runs/20260424_151340_32k_singlewrite_baseline_llama1b`
- 32K no promotion: `phase2_runs/20260424_150933_32k_dualwrite_nopromotion_llama1b`
- 32K promotion: `phase2_runs/20260424_150759_32k_dualwrite_promotion_llama1b`
- 32K oracle: `phase2_runs/20260424_151040_32k_oracle_llama1b`

## Remaining Work
- Repeat the Qwen2.5-7B 32K matrix if variance bars are needed for the paper.
- Replace the request-driven promotion probe with a true independent background scheduler for non-layerwise and layerwise paths.
- Add layerwise disk retrieve write-back or a direct promotion API so layerwise runs no longer need a promotion proxy.
