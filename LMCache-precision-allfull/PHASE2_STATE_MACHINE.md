# Phase 2 State Machine

## States
- `MISS`
- `BASE_READY`
- `PROMOTING`
- `FULL_READY`
- `FULL_EVICTED` (documented boundary, not yet implemented as an LMCache eviction hook)

## Implemented Bootstrap Transitions
```text
MISS
  -> BASE_READY        base namespace primed
BASE_READY
  -> PROMOTING         hotness threshold reached and promotion enqueued
PROMOTING
  -> FULL_READY        full-materialization proxy completed
PROMOTING
  -> BASE_READY        promotion failed
FULL_READY
  -> FULL_READY        promoted follow-up reused full namespace
FULL_READY
  -> FULL_EVICTED      full fast-tier copy evicted after base cleanup
FULL_EVICTED
  -> FULL_READY        full reloaded from slow tier and written back to fast tier
FULL_EVICTED
  -> MISS              slow-tier full is also missing or reload fails
```

## State Semantics
- `MISS`: no usable base or full cache is known for the prefix.
- `BASE_READY`: base is available in the fast tier and full is available in the slow tier.
- `PROMOTING`: base remains available in the fast tier while full is being materialized from the slow tier.
- `FULL_READY`: full is available in the fast tier. When `cleanup_base_on_full_store=True`, the base fast-tier copy may already have been removed.
- `FULL_EVICTED`: full was previously promoted but the fast-tier full copy was evicted. Because base may have been cleaned up after promotion, this state must not be treated as `BASE_READY`.

## Counters
- `base_hit_count`: increments when a follow-up is served from `BASE_READY`.
- `full_hit_count`: increments when a follow-up is served from `FULL_READY`.
- `full_reuse_count`: increments only for post-promotion `FULL_READY` hits.
- `promotion_enqueued`: increments when `BASE_READY -> PROMOTING` is accepted.
- `promotion_success`: increments when `PROMOTING -> FULL_READY` succeeds.
- `promotion_failure`: increments when promotion rolls back to `BASE_READY`.

## Promotion Policy
The bootstrap policy is deliberately simple: enqueue promotion after `promotion_min_access_count` base hits. The default is `1`, so the first measured base hit promotes the prefix for later follow-up requests.

## FULL_EVICTED Handling
`FULL_EVICTED` is the correct fallback state when a promoted full copy leaves the fast tier after base cleanup. In this state the system should prefer the slow-tier full copy rather than recomputing from scratch:

```python
if state == "FULL_EVICTED":
    # Slow tier should still contain the full variant from dual-store.
    # Retrieve full from slow tier, optionally write it back to fast tier,
    # then transition to FULL_READY if retrieval succeeds.
    return "full_from_slow"
```

If the slow-tier full copy is also missing, the state should fall back to `MISS`, not `BASE_READY`. This distinction matters because promotion cleanup may already have removed the base fast-tier copy.

Current implementation boundary: Phase2 harness records `BASE_READY`, `PROMOTING`, and `FULL_READY`; LMCache does not yet expose an eviction callback that transitions `FULL_READY -> FULL_EVICTED`.
