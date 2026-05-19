# Scheduler Overlap Feasibility Analysis (2026-05-05)

## Goal

We want to reduce GPU idle time caused by full-KV retrieve/writeback. The proposed simple idea is:

- when one request is waiting for KV transfer,
- let vLLM scheduler schedule queued MISS requests for normal prefill,
- so GPU can do compute while the other request waits on I/O.

## Backup

Before touching scheduler-related code, the current server worktree was archived:

- Archive: `/home/panzihang/lmcache_code_backups/LMCache-phase0-codex_code_before_scheduler_overlap_20260505_175511.tar.gz`
- SHA256: `7819e20ecdd2d88a690cba7d0199da698d1ba35b8996e3759b4fe8b124be388e`
- Scope: source/docs/scripts/reports/modlogs, excluding `phase2_runs` and cache directories.

## Finding 1: vLLM scheduler already has the proposed waiting-request skip logic

The active scheduler is:

`/home/panzihang/venvs/vllm-stable/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py`

Relevant behavior:

- `schedule()` scans `self.waiting` and `self.skipped_waiting`.
- If a request is in `RequestStatus.WAITING_FOR_REMOTE_KVS` and not finished, scheduler moves it into `step_skipped_waiting` and continues the loop.
- That means other waiting requests can be scheduled in the same scheduler step.

Relevant lines in this vLLM build:

- `scheduler.py:563-590`: waiting queue loop and blocked-status skip.
- `scheduler.py:787-795`: async KV load requests become `WAITING_FOR_REMOTE_KVS` and are put back into skipped waiting.
- `scheduler.py:2042-2093`: once `finished_recving` arrives, request moves back to `WAITING` or `PREEMPTED`.
- `scheduler.py:2111-2131`: worker-side connector output populates `finished_recving_kv_req_ids`.

So the simple scheduler-side policy is not missing from vLLM; it is already present for connectors that use vLLM's async KV-transfer protocol.

## Finding 2: LMCache non-layerwise path does not enter vLLM async KV-transfer protocol

The installed vLLM wrapper is:

`/home/panzihang/venvs/vllm-stable/lib/python3.10/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`

Its scheduler-side method returns:

```python
return self._lmcache_engine.get_num_new_matched_tokens(
    request, num_computed_tokens
), False
```

The second value is `load_kv_async`, and it is hard-coded to `False`.

That means LMCache requests are scheduled as normal running requests, not as `WAITING_FOR_REMOTE_KVS`. Therefore, vLLM never gets a chance to schedule a MISS request while that request is waiting for LMCache retrieve.

## Finding 3: Forcing `load_kv_async=True` would likely deadlock

The LMCache worker implementation is:

`/home/panzihang/src/LMCache-phase0-codex/lmcache/integration/vllm/vllm_v1_adapter.py`

Relevant behavior:

- `get_num_new_matched_tokens()` only reports how many tokens are externally cached.
- `update_state_after_alloc()` marks the request's `LoadSpec.can_load=True`.
- actual non-layerwise KV load happens inside worker-side `start_load_kv()` by calling `lmcache_engine.retrieve(...)`.
- `get_finished()` currently returns `(None, None)`.

This matters because if scheduler is changed to put LMCache loads into `WAITING_FOR_REMOTE_KVS`, the request will not be included in `scheduler_output.scheduled_new_reqs`, so worker-side `start_load_kv()` will not run for it. Since `get_finished()` never reports `finished_recving`, the request can remain stuck forever.

So a scheduler-only behavior patch is unsafe.

## Interpretation

The performance issue is real, but the missing piece is not merely vLLM scheduler queue policy. The missing piece is an LMCache async load implementation that can:

1. start KV transfer without entering the normal forward pass,
2. write into allocated GPU KV blocks safely,
3. report `finished_recving` to vLLM,
4. then let scheduler promote the request from `WAITING_FOR_REMOTE_KVS` back to `WAITING`.

Without those four pieces, changing only `schedule()` is likely to hang or corrupt request state.

## Safe Next Step

Do not patch scheduler behavior directly yet.

The low-risk next step is an experiment/diagnostic patch that measures the opportunity:

- count how often LMCache full-hit requests block inside non-layerwise `start_load_kv()`;
- measure retrieve time and GPU idle window;
- count how many queued MISS requests existed during that window.

If this opportunity is large, the real implementation should be one of:

1. fix layerwise prefetch path and fake_lookup_id boundary, then use layerwise pipelining;
2. implement a true LMCache async KV-load connector that returns `load_kv_async=True` and reports `finished_recving`;
3. build a request-level two-stage scheduler where KV load is a separate worker task before vLLM schedules compute.

## Current Decision

No scheduler behavior change was applied after backup, because the proposed scheduler-only patch is already present in vLLM and LMCache cannot safely use it yet.
