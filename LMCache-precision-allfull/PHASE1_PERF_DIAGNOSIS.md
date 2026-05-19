# Phase 1 Performance Diagnosis

## Machine-readable Inputs
- codec benchmark: `/home/panzihang/src/LMCache-phase0-codex/phase1_runs/20260423_145714_int8_codec_cost/int8_codec_cost.json`
- summary: `/home/panzihang/src/LMCache-phase0-codex/PHASE1_RESULTS_SUMMARY.json`

## Key Findings
1. Pure `Int8BaseCodec` cost is already material at request scale.
   - `16K O1 boundary`: encode `198.967 ms`, decode `107.849 ms`
   - `16K O2 boundary`: encode `573.803 ms`, decode `231.057 ms`
   - `32512 O1 boundary`: encode `721.847 ms`, decode `287.880 ms`
   - `32512 O2 boundary`: encode `1012.685 ms`, decode `461.924 ms`
2. This means the slowdown is not a vague `python_fallback` effect; it is specifically dominated by Python-side boundary `encode/decode`, especially on the layerwise `O2` path.
3. But codec math alone still does **not** explain all of `O2 medium`.
   - `16K medium O2 store delta vs full`: `2721.128 ms`
   - scaled codec encode estimate for the stored half: `286.901 ms`
   - therefore most of the remaining penalty is elsewhere in the layerwise store path.
4. The current `full-hit` `O2 high` path still cannot use true layerwise overlap.
   - `medium` runs emit adapter-level `Retrieved N tokens` from `wait_for_layer_load()`
   - `high full-hit` runs do not; they only emit the eager-drain warning and final cache-engine retrieve log
   - therefore the current vLLM hook sequence does not enter `wait_for_layer_load()` for that full-hit path

## Practical Conclusion
- `codec` quality is fixed
- `codec` cost is large enough that Python-side optimization is not optional
- `O2 high` cannot be made into a real overlap path by only editing `wait_for_layer_load()`; current full-hit behavior needs a different synchronization hook or a different retrieval strategy
- `O2 medium` also has significant non-codec layerwise overhead on top of codec cost
