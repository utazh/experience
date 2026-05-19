# Paper Caveats 2026-05-04

This note records the system boundaries that must be stated accurately in the paper draft.

## Terminology

- The base codec uses per-vector symmetric INT8 quantization. For each KV vector, it computes one scale with `amax(dim=-1)`, quantizes that vector to INT8, and stores the scale separately.
- Do not describe the current codec as per-channel quantization.

## Implemented And Validated

- Base/full storage routing has been smoke-tested for the Phase 2 path: base goes to `LocalCPUBackend`, full goes to `LocalDiskBackend`, and no base files appear in the disk tier.
- Core auto policy can select fidelity from a state-store-style input: `BASE_READY` and `PROMOTING` choose base, while `FULL_READY` chooses full.
- Base cleanup after full materialization is implemented for the non-layerwise path and has been validated in 16K Phase 2 logs.
- 16K hot-prefix Phase 2 results are valid for the current request-driven promotion harness.

## Required Caveats

- Promotion is request-driven in the current experiments. A later request materializes full KV from the slow tier and writes it back to the fast tier; there is not yet an independent background promotion scheduler.
- `FULL_EVICTED` is documented as a required state, but there is no eviction hook that automatically changes `FULL_READY` to `FULL_EVICTED` in the running LMCache engine.
- Base cleanup currently depends on the non-layerwise full retrieve/write-back semantics. Layerwise promotion cleanup remains future work because there is no explicit full fast-tier write-back barrier on that path.
- Prefix state is in-memory only. Persistence across process restarts and synchronization across multiple LMCache instances are future work.
- The internal state-store path introduced in this patch is experimental and must be compared against the earlier harness-state 16K results before the paper claims a fully closed-loop precision scheduler.

## Recommended Paper Wording

- Safe wording now: "LMCache maintains or receives a prefix fidelity state and uses a core auto policy to select the appropriate KV precision."
- Wording allowed only after internal-state comparison passes: "LMCache maintains prefix fidelity state internally and uses it to automatically select base or full KV precision on the request path."
- Avoid: "background promotion scheduler" unless a real scheduler is implemented and measured.
- Avoid: "per-channel INT8" for the current codec.

## Validation Still Needed

- Run a harness-state 16K Phase 2 control and an internal-state 16K Phase 2 run under the same settings.
- Treat the internal-state path as correct if the no-promotion and promotion TTFT numbers stay within roughly 5-10% of the harness-state control and the raw logs show `PHASE2_INTERNAL_STATE_*` plus `PHASE2_CORE_AUTO_DECISION` events.
- After that comparison, proceed to ShareGPT, HotpotQA, and QA/F1/EM experiments.
