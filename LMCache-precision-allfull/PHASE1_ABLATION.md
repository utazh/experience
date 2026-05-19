# Phase 1 Ablation

## Reference Runs
- Phase 0 baseline: `/home/panzihang/src/LMCache-phase0-codex/phase0_runs/20260422_203840`
- Phase 1 full reference: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260422_235841_full_reference`
- Phase 1 O1 (`base-only-no-prefetch`, new codec): `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260423_141255_o1_base_no_prefetch_new_codec_rerun`
- Phase 1 O2 (`base-prefetch`, new codec): `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260423_141255_o2_base_prefetch_new_codec_rerun`
- `4K high reuse, max_tokens=16` smoke:
  - full: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260423_140954_full_ref_4k_high_mt16`
  - O2: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260423_140954_o2_4k_high_mt16_new_codec`
- Int8 codec round-trip JSON: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260423_142130_int8_codec_roundtrip/int8_codec_roundtrip.json`
- consolidated summary: `/home/panzihang/src/LMCache-phase0-codex/PHASE1_RESULTS_SUMMARY.csv`

## Main Results

| context | reuse | baseline ttft_ms | full-ref ttft_ms | O1 ttft_ms | O2 ttft_ms | O2 - O1 (ms) | O1 match | O2 match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16384 | medium | 2661.187 | 2512.836 | 3032.839 | 3028.194 | -4.645 | 1.0 | 1.0 |
| 16384 | high | 399.969 | 267.089 | 490.565 | 710.740 | +220.175 | 1.0 | 1.0 |
| 32512 | medium | 6660.682 | 6310.750 | 7181.633 | 7340.292 | +158.659 | 1.0 | 1.0 |
| 32512 | high | 630.572 | 620.720 | 1019.588 | 1406.015 | +386.427 | 1.0 | 1.0 |

## Interpretation
1. The new per-vector `Int8BaseCodec` restores quality in all main Phase 1 scenarios. `output_match_rate_vs_full_ref = 1.0` now holds for both `medium` and `high` reuse at `16K` and `32512`.
2. The `4K high reuse, max_tokens=16` smoke matched the full-reference token sequence exactly, so this is not a one-token coincidence.
3. Base KV still reduces retrieved bytes to about `0.5x` of full-reference, so the compression target is working as intended.
4. Under `python_fallback=true`, the new boundary `encode/decode` cost is large enough that TTFT is now worse than both the full-reference run and the Phase 0 baseline in every `16K/32512` `medium/high` row.
5. The O1 vs O2 ablation is now much clearer in the negative direction: `base-prefetch` does not beat `base-only` under the current Python path. The only positive delta is a marginal `-4.645 ms` at `16K medium`; every other main row is slower with O2.
6. This means the overlap benefit is currently masked by two costs:
   - Python-side per-vector quantize/dequantize on the critical path
   - the Phase 1 `full-hit + layerwise` eager-drain fallback, which is correctness-safe for measurement but does not preserve the intended overlap pipeline

## Immediate Conclusion
- Engineering status: `Go`
- Quality status: `Go`
- Performance status under Python fallback: `No-Go`
- Phase 1 now proves that `base KV + real int8 codec` can preserve quality, but it does **not** yet improve TTFT in the current fallback implementation.
