# Phase 2 Promotion Architecture

## Initial Experiment Boundary
This Phase 2 start point is a remote-only experiment harness. It does not modify the local workspace.

The current Phase 1 implementation has request-level `full` and `base` namespaces, but it does not yet materialize both variants from one foreground request despite exposing `store_full_variant` and `store_base_variant` config fields. Because of that, the first Phase 2 harness keeps the promotion state machine at the benchmark layer and uses a `full_materialization_proxy` request to emulate the background action that makes a prefix `FULL_READY`.

This is intentionally a bootstrap experiment, not the final core implementation.

## Components
- `benchmarks/quality_aware_kv/phase2/run_phase2_hot_prefix.py`
  - Builds high-reuse hot-prefix workloads.
  - Drives `no_promotion`, `promotion`, and `oracle_full_ready` scenarios in one vLLM engine.
  - Emits `PHASE2_CONFIG`, `PHASE2_REQ`, `PHASE2_STATE`, and `PHASE2_SUMMARY` JSON markers.
- `benchmarks/quality_aware_kv/phase2/parse_phase2_hot_prefix.py`
  - Parses request rows, state transitions, LMCache retrieve/store logs, and summary metrics.

## Scenario Semantics
- `no_promotion`: prime the base namespace, then keep serving follow-up requests from `BASE_READY`.
- `promotion`: prime the base namespace, serve the first follow-up from `BASE_READY`, enqueue promotion, materialize the full namespace, then serve later follow-ups from `FULL_READY`.
- `oracle_full_ready`: prime the full namespace before follow-up requests. This is an upper-bound reference.

## Metrics
- TTFT for first and later follow-up requests.
- `output_match_rate_vs_full_ref` relative to a skip-save full reference.
- `promotion_enqueued`, `promotion_success`, `promotion_failure`.
- `full_reuse_count` and `promotion_reuse_rate`.
- LMCache retrieve/store size and latency parsed from logs when available.

## Known Caveat
`promotion` currently includes a foreground full-materialization proxy. The final Phase 2 core should move this below the benchmark layer: full should already exist in a slower namespace or tier, and promotion should warm it into the target hot layer without recomputing the prefix on the foreground path.
