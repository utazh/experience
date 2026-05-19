# Phase 1 Architecture

## Scope
Phase 1 was brought up under:
- `python_fallback=true`
- `c_ops_available=false`
- main target regime: `Qwen2.5-7B-Instruct`, `16K` and `32512`, `medium/high` reuse

## Implemented Pieces
1. Request-level fidelity namespace
   - added `full/base/auto` normalization in `lmcache/v1/fidelity/*`
   - request configs now carry `lmcache.fidelity`, `lmcache.fidelity_policy`, `lmcache.fidelity_reason`
2. Config surface
   - extended `lmcache/v1/config.py` with fidelity/cache policy fields
3. Cache-engine integration
   - `store`, `retrieve`, `retrieve_layer`, `lookup`, and async lookup paths now normalize fidelity configs
   - base requests now use boundary encode/decode instead of raw dtype downcast
4. Real Int8 base codec
   - `Int8BaseCodec` now performs per-vector symmetric quantization along the last hidden dimension
   - store path writes `int8 quantized tensor + float32 scale tensor`
   - retrieve path reconstructs the original floating KV dtype before H2D scatter
   - codec metadata is carried in `MemoryObjMetadata.extra`
5. Storage-path compatibility
   - `StorageManager.allocate_and_copy_objects()` now copies grouped/raw tensors so encoded `MemoryObj`s remain portable inside LMCache
6. vLLM adapter wiring
   - layerwise store/retrieve pass request configs and req_id into LMCache engine
   - for `full-hit + layerwise + vllm_cached_tokens=0`, adapter still uses an eager-drain fallback on the existing `retrieve_layer()` generator so all per-layer loads finish, synchronize, and emit machine-readable stats even when vLLM does not call `wait_for_layer_load()`
7. Validation toolchain
   - `benchmarks/quality_aware_kv/phase1/run_phase1_matrix.py`
   - `benchmarks/quality_aware_kv/phase1/parse_phase1_matrix.py`
   - `benchmarks/quality_aware_kv/phase1/summarize_phase1_results.py`
   - `benchmarks/quality_aware_kv/phase1/test_int8_codec_roundtrip.py`

## Current Base Codec Semantics
- `FakeBaseCodec`: pipeline validation only
- `Int8BaseCodec`: real per-vector symmetric quantizer/dequantizer
- round-trip validation on CPU currently reports:
  - `overall_max_err = 0.015625`
  - `compression_ratio ~= 0.508`
- smoke validation on `4K high reuse, max_tokens=16` shows exact token-level agreement with full reference

## Important Diagnostic Findings
1. The original `high reuse` quality failure was caused by the placeholder base codec, not by result parsing.
2. The new codec fixes the quality issue, but the Python fallback implementation shifts the bottleneck into CPU-side encode/decode and layerwise synchronization overhead.
3. The previous `O2 high` logging hole was repaired with an eager-drain narrow fix; this is appropriate for Phase 1 measurement, but it is not the final overlap design.

## Known Gaps
1. Boundary encode/decode is still Python-side and sits on the critical path for both store and retrieve.
2. `O2` still uses the eager-drain `full-hit` fallback for correctness-safe measurement; a proper overlap-friendly synchronization design is still needed.
3. `c_ops` remains out of scope for this thread.
