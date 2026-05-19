# Phase 2 Cleanup + E2E Results (2026-05-03)

## Scope
All modifications were made on the remote server under `/home/panzihang/src/LMCache-phase0-codex`.
Baseline copies and the full change log are in `_codex_modlog/20260503_cleanup_e2e/`.

## Modified Files
- `lmcache/v1/cache_engine.py`
- `lmcache/v1/config.py`
- `benchmarks/quality_aware_kv/phase2/run_phase2_hot_prefix.py`
- `benchmarks/quality_aware_kv/phase2/parse_phase2_hot_prefix.py`
- `benchmarks/quality_aware_kv/phase2/verify_phase2_storage_routing.py`
- `lmcache/c_ops.py -> lmcache/c_ops_python_fallback.py`

## What Changed
- Archived old backup-like files into `_codex_modlog/20260503_cleanup_e2e/archived_backups/` instead of deleting them.
- Renamed `lmcache/c_ops.py` to `lmcache/c_ops_python_fallback.py`; native `lmcache/c_ops.cpython-310-x86_64-linux-gnu.so` remains the import target.
- Added `cleanup_base_on_full_store` / `LMCACHE_CLEANUP_BASE_ON_FULL_STORE`, default `False`.
- Added `PHASE2_BASE_CLEANUP` logging and cleanup on full store plus full retrieve/writeback paths.
- Added `PHASE2_AUTO_DECISION` logging for the harness-level auto policy.
- Added `verify_phase2_storage_routing.py` to check base/full storage-layer routing from logs and disk files.
- Extended the Phase2 parser to extract dual-write sync split and base-cleanup rows.
- Hardening patch `_codex_modlog/20260503_hardening_patch/`: Phase2 dual-store now fail-fasts when the configured backend is unavailable; base dual-store target is constrained to `LocalCPUBackend`; layerwise base cleanup is skipped with `PHASE2_BASE_CLEANUP_SKIPPED` until there is an explicit full fast-tier writeback barrier.
- Parser/mixed patch `_codex_modlog/20260503_parser_mixed_core_auto/`: parser now records `PHASE2_CORE_AUTO_DECISION` rows and annotates core-auto requests with match fields; `run_phase2_hot_cold_mixed.py` now uses the same core-auto state metadata path as hot-prefix followups.

## Validation
- Native `c_ops.so` import validated: `python_fallback=false`.
- Existing 512-token route smoke passed: base logged as `LocalCPUBackend`, full logged as `LocalDiskBackend`, disk had only `fidelity%full` files.
- Cleanup retry smoke: `phase2_runs/20260503_144506_cleanup_auto_smoke_512_llama1b_retry`
- Cleanup smoke final state: `FULL_READY`, promotion_success=1, full_reuse_count=2, mean_followup_match_rate=1.0.
- Base cleanup removed 2 chunks at promotion materialization; later full hits removed 0 as expected.
- Hardening smoke: `phase2_runs/20260503_hardening_smoke_512_llama1b`.
- Hardening smoke final state: `FULL_READY`, promotion_success=1, full_reuse_count=1, mean_followup_match_rate=1.0.
- Hardening routing check passed: 1 `PHASE2_DUAL_WRITE` event, base logged as `LocalCPUBackend`, full logged as `LocalDiskBackend`, disk had 2 full files and 0 base files.
- Fail-fast unit checks passed: missing configured backend raises `ValueError`; `base -> LocalDiskBackend` is rejected before being treated as a supported experiment path.
- Core auto policy patch `_codex_modlog/20260503_core_auto_policy/`: `policy.py` now supports `auto_state_store_v0`; Phase2 followup requests pass state metadata and do not set explicit `lmcache.fidelity`, so the selected base/full fidelity comes from the LMCache normalization/policy path.
- Core auto smoke: `phase2_runs/20260503_core_auto_smoke_512_llama1b_retry`. Followup 1 used `BASE_READY -> base` through `core_auto_state_store_v0`; followup 2 used `FULL_READY -> full` through `core_auto_state_store_v0`.
- Core auto smoke final state: `FULL_READY`, promotion_success=1, full_reuse_count=1, mean_followup_match_rate=1.0. Routing check passed: base `LocalCPUBackend`, full `LocalDiskBackend`, disk had 2 full files and 0 base files.
- Mixed core-auto smoke: `phase2_runs/20260503_mixed_core_auto_smoke_512_llama1b`. Parser recorded 6 `PHASE2_CORE_AUTO_DECISION` entries; parsed followups all had `core_auto_match=true` for cold `BASE_READY -> base`, hot `BASE_READY -> base`, and hot `FULL_READY -> full`. Routing check passed with 2 dual-write events, 4 full disk files, and 0 base disk files.

## Write Amplification Split
See `PHASE2_WRITE_AMP_SYNC_SPLIT_20260503.md` and `.json`.
- Qwen2.5-7B 32760-token first-request TTFT overhead: `641.333 ms` (`5.917%`).
- Explicit dual-write sync section: `1428.344 ms`, dominated by base encode `1418.4621000000002 ms`; full disk submit was `8.971599999999999 ms`.

## Known Boundaries
- LMCACHE_CLEANUP_BASE_ON_FULL_STORE defaults to False and must be explicitly enabled for cleanup experiments.
- The current auto state source is still harness-level state_store_v0, but fidelity selection for patched Phase2 followups runs through core `lmcache/v1/fidelity/policy.py` as `auto_state_store_v0`. `PHASE2_AUTO_DECISION` records harness state input; `PHASE2_CORE_AUTO_DECISION` records core policy output.
- Base int8 persistence is an experiment constraint, not a generalized storage feature: dual-store base must target `LocalCPUBackend`; `LocalDiskBackend` does not persist the codec metadata needed by the int8 base variant.
- Base cleanup is only active on non-layerwise full materialization/reuse paths. Layerwise cleanup is intentionally skipped because that path does not currently prove full is written back into the fast tier.
- Phase2 auto state is still supplied by the experiment harness as `state_store_v0` metadata; the fidelity decision now runs through core `lmcache/v1/fidelity/policy.py`, but the state store itself is not yet a persistent LMCache service.
- FULL_READY eviction fallback is not fully implemented as an LMCache eviction hook. With base cleanup enabled, downgrading to BASE_READY is not generally correct because the base fast-tier copy was removed; a future robust state model should distinguish FULL_READY_FAST, FULL_READY_SLOW, BASE_READY, and MISS or verify fast-tier full before choosing full.
- Promotion is still request-driven full-disk retrieve/writeback, not an independent background scheduler, per current experiment scope.
