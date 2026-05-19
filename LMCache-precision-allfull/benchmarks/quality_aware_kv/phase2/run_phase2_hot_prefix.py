#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import lmcache.c_ops as lmc_ops
from transformers import AutoTokenizer
from vllm import TokensPrompt
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

STATE_STORE_V0_STATE_KEY = "lmcache.state_store_v0.state"
STATE_STORE_V0_PREFIX_ID_KEY = "lmcache.state_store_v0.prefix_id"
STATE_STORE_V0_SEQUENCE_INDEX_KEY = "lmcache.state_store_v0.sequence_index"
STATE_STORE_V0_ENABLE_PROMOTION_KEY = "lmcache.state_store_v0.enable_promotion"


class PrefixState(str, Enum):
    MISS = "MISS"
    BASE_READY = "BASE_READY"
    PROMOTING = "PROMOTING"
    FULL_READY = "FULL_READY"


@dataclass
class PrefixStats:
    access_count: int = 0
    base_hit_count: int = 0
    full_hit_count: int = 0
    full_reuse_count: int = 0
    promotion_enqueued: int = 0
    promotion_success: int = 0
    promotion_failure: int = 0
    promotion_dropped: int = 0
    promotion_latency_ms: list[float] = field(default_factory=list)


class Phase2StateMachine:
    def __init__(self, prefix_id: str, scenario: str):
        self.prefix_id = prefix_id
        self.scenario = scenario
        self.state = PrefixState.MISS
        self.stats = PrefixStats()

    def _emit(self, event: str, before: PrefixState, after: PrefixState, **extra: Any) -> None:
        row = {
            "event": event,
            "scenario": self.scenario,
            "prefix_id": self.prefix_id,
            "state_before": before.value,
            "state_after": after.value,
        }
        row.update(extra)
        print("PHASE2_STATE " + json.dumps(row, ensure_ascii=False), flush=True)

    def transition(self, after: PrefixState, event: str, **extra: Any) -> None:
        before = self.state
        self.state = after
        self._emit(event, before, after, **extra)

    def mark_base_ready(self, reason: str) -> None:
        if self.state == PrefixState.MISS:
            self.transition(PrefixState.BASE_READY, "MISS_TO_BASE_READY", reason=reason)
        elif self.state == PrefixState.BASE_READY:
            self._emit("BASE_READY_STABLE", self.state, self.state, reason=reason)

    def mark_full_ready(self, reason: str, promotion_latency_ms: float | None = None) -> None:
        if promotion_latency_ms is not None:
            self.stats.promotion_latency_ms.append(promotion_latency_ms)
        self.transition(PrefixState.FULL_READY, "PROMOTING_TO_FULL_READY", reason=reason, promotion_latency_ms=promotion_latency_ms)

    def choose_auto_fidelity(self, sequence_index: int, state_source: str = "harness") -> tuple[str, str]:
        if self.state == PrefixState.FULL_READY:
            fidelity = "full"
            reason = "full_ready_hit"
        elif self.state == PrefixState.PROMOTING:
            fidelity = "base"
            reason = "promoting_read_base_no_duplicate"
        else:
            fidelity = "base"
            reason = "base_ready_hit"
        print(
            "PHASE2_AUTO_DECISION "
            + json.dumps(
                {
                    "event": "auto_policy_decision",
                    "scenario": self.scenario,
                    "prefix_id": self.prefix_id,
                    "sequence_index": sequence_index,
                    "state": self.state.value,
                    "selected_fidelity": fidelity,
                    "reason": reason,
                    "decision_source": "harness_state_store_v0_input_for_core_auto"
                    if state_source == "harness"
                    else "harness_expected_label_internal_state_source",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return fidelity, reason

    def observe_base_hit(self, min_access_count: int, enable_promotion: bool) -> bool:
        self.stats.access_count += 1
        self.stats.base_hit_count += 1
        if not enable_promotion:
            return False
        if self.state == PrefixState.PROMOTING:
            self.stats.promotion_dropped += 1
            self._emit(
                "PROMOTION_DROP_STATE",
                self.state,
                self.state,
                reason="already_promoting",
            )
            return False
        if self.state != PrefixState.BASE_READY:
            self.stats.promotion_dropped += 1
            self._emit("PROMOTION_DROP_STATE", self.state, self.state, reason="not_base_ready")
            return False
        if self.stats.access_count < min_access_count:
            return False
        self.stats.promotion_enqueued += 1
        self.transition(
            PrefixState.PROMOTING,
            "BASE_READY_TO_PROMOTING",
            reason="hotness_threshold",
            access_count=self.stats.access_count,
            promotion_queue_length=1,
        )
        return True

    def observe_full_hit(self, promoted: bool) -> None:
        self.stats.access_count += 1
        self.stats.full_hit_count += 1
        if promoted:
            self.stats.full_reuse_count += 1
        self._emit(
            "FULL_READY_REUSE" if promoted else "FULL_READY_ORACLE_HIT",
            self.state,
            self.state,
            full_reuse_count=self.stats.full_reuse_count,
        )

    def promotion_failed(self, reason: str) -> None:
        self.stats.promotion_failure += 1
        self.transition(PrefixState.BASE_READY, "PROMOTING_TO_BASE_READY", reason=reason)

    def summary_dict(self) -> dict[str, Any]:
        promotion_latency_p50_ms = None
        if self.stats.promotion_latency_ms:
            promotion_latency_p50_ms = statistics.median(self.stats.promotion_latency_ms)
        reuse_rate = None
        if self.stats.promotion_success:
            reuse_rate = self.stats.full_reuse_count / self.stats.promotion_success
        return {
            "scenario": self.scenario,
            "prefix_id": self.prefix_id,
            "final_state": self.state.value,
            "access_count": self.stats.access_count,
            "base_hit_count": self.stats.base_hit_count,
            "full_hit_count": self.stats.full_hit_count,
            "full_reuse_count": self.stats.full_reuse_count,
            "promotion_enqueued": self.stats.promotion_enqueued,
            "promotion_success": self.stats.promotion_success,
            "promotion_failure": self.stats.promotion_failure,
            "promotion_dropped": self.stats.promotion_dropped,
            "promotion_latency_p50_ms": promotion_latency_p50_ms,
            "promotion_reuse_rate": reuse_rate,
        }


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def build_token_pool(tokenizer, prefix: str, n_words: int) -> list[int]:
    text = " ".join(f"{prefix}_{i}" for i in range(n_words))
    return tokenizer.encode(text, add_special_tokens=False)


def prefix_id_for(tokens: list[int]) -> str:
    digest = hashlib.sha1(",".join(str(t) for t in tokens).encode("utf-8")).hexdigest()
    return digest[:16]


def output_match_rate(reference: list[int] | None, candidate: list[int]) -> float | None:
    if not reference:
        return None
    matches = 0
    for idx, token_id in enumerate(reference):
        if idx < len(candidate) and candidate[idx] == token_id:
            matches += 1
    return matches / len(reference)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def current_promotion_mode(use_layerwise: bool) -> str:
    dual_write_active = env_bool("LMCACHE_STORE_BASE_VARIANT") and env_bool(
        "LMCACHE_STORE_FULL_VARIANT", True
    )
    full_target = os.environ.get("LMCACHE_STORE_FULL_TARGET", "local_cpu").lower()
    full_to_disk = full_target in {"local_disk", "localdisk", "disk", "localdiskbackend"}
    if dual_write_active and full_to_disk and not use_layerwise:
        return "request_driven_full_disk_retrieve_writeback"
    if dual_write_active:
        return "dual_write_enabled_request_proxy"
    return "full_materialization_proxy"


def current_experiment_boundary(use_layerwise: bool) -> str:
    promotion_mode = current_promotion_mode(use_layerwise)
    if promotion_mode == "request_driven_full_disk_retrieve_writeback":
        return (
            "first-request base/full dual-store is enabled; base is stored to "
            "the configured fast target and full to the configured slow target; "
            "promotion is approximated by a request-driven full-disk retrieve that "
            "writes back to LocalCPU, not by an independent background scheduler"
        )
    if promotion_mode == "dual_write_enabled_request_proxy":
        return (
            "first-request base/full dual-store is enabled; this layerwise harness "
            "still uses a request-driven full-materialization/promotion proxy because "
            "layerwise disk retrieve does not yet perform LocalCPU write-back"
        )
    return (
        "script-level state machine; promotion uses a full-materialization proxy "
        "because the run did not enable first-request dual-store full/base variants"
    )


async def run_request(
    engine: AsyncLLM,
    req_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    fidelity: str | None,
    skip_save: bool = False,
    core_auto_state: str | None = None,
    prefix_id: str | None = None,
    sequence_index: int | None = None,
    enable_internal_promotion: bool | None = None,
) -> dict[str, Any]:
    kv_params: dict[str, Any] = {}
    if fidelity is not None:
        kv_params["lmcache.fidelity"] = fidelity
    if core_auto_state is not None:
        kv_params[STATE_STORE_V0_STATE_KEY] = core_auto_state
    if prefix_id is not None:
        kv_params[STATE_STORE_V0_PREFIX_ID_KEY] = prefix_id
    if sequence_index is not None:
        kv_params[STATE_STORE_V0_SEQUENCE_INDEX_KEY] = sequence_index
    if enable_internal_promotion is not None:
        kv_params[STATE_STORE_V0_ENABLE_PROMOTION_KEY] = enable_internal_promotion
    if skip_save:
        kv_params["lmcache.skip_save"] = True
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        output_kind=RequestOutputKind.DELTA,
        extra_args={"kv_transfer_params": kv_params},
    )
    start = time.perf_counter()
    start_wall_ts = now_str()
    first = None
    first_wall_ts = None
    final_text = ""
    output_token_ids: list[int] = []
    engine_req_id = req_id
    async for out in engine.generate(
        TokensPrompt(prompt_token_ids=prompt_token_ids),
        params,
        request_id=req_id,
    ):
        if first is None:
            first = time.perf_counter()
            first_wall_ts = now_str()
        engine_req_id = getattr(out, "request_id", engine_req_id)
        if out.outputs:
            sample = out.outputs[0]
            final_text += sample.text
            delta_token_ids = getattr(sample, "token_ids", None)
            if delta_token_ids:
                output_token_ids.extend(list(delta_token_ids))
    end = time.perf_counter()
    return {
        "req_id": req_id,
        "engine_req_id": engine_req_id,
        "start_wall_ts": start_wall_ts,
        "first_token_wall_ts": first_wall_ts,
        "end_wall_ts": now_str(),
        "ttft_ms": None if first is None else round((first - start) * 1000.0, 3),
        "e2e_ms": round((end - start) * 1000.0, 3),
        "output_preview": final_text[:160],
        "output_token_ids": output_token_ids,
    }


async def emit_request(
    engine: AsyncLLM,
    scenario: str,
    prefix_id: str,
    request_kind: str,
    sequence_index: int,
    state_before: PrefixState,
    state_after: PrefixState,
    prompt_tokens: list[int],
    max_tokens: int,
    fidelity: str,
    reference_output_token_ids: list[int] | None,
    decision_reason: str,
    skip_save: bool = False,
    use_core_auto: bool = False,
    state_source: str = "harness",
    enable_internal_promotion: bool | None = None,
) -> dict[str, Any]:
    req_id = f"phase2_{scenario}_{request_kind}_{sequence_index}_{fidelity}_{prefix_id}"
    result = await run_request(
        engine,
        req_id,
        prompt_tokens,
        max_tokens,
        None if use_core_auto else fidelity,
        skip_save=skip_save,
        core_auto_state=state_before.value
        if use_core_auto and state_source == "harness"
        else None,
        prefix_id=prefix_id if use_core_auto else None,
        sequence_index=sequence_index if use_core_auto else None,
        enable_internal_promotion=enable_internal_promotion,
    )
    row = {
        "event": "request_result",
        "scenario": scenario,
        "prefix_id": prefix_id,
        "request_kind": request_kind,
        "sequence_index": sequence_index,
        "state_before": state_before.value,
        "state_after": state_after.value,
        "requested_fidelity": fidelity,
        "decision_reason": decision_reason,
        "fidelity_policy_path": (
            "core_auto_internal_state_store_v0"
            if use_core_auto and state_source == "internal"
            else "core_auto_state_store_v0"
            if use_core_auto
            else "explicit_request"
        ),
        "state_source": state_source,
        "prompt_tokens": len(prompt_tokens),
        "skip_save": skip_save,
        "output_match_rate_vs_full_ref": output_match_rate(reference_output_token_ids, result["output_token_ids"]),
    }
    row.update(result)
    print("PHASE2_REQ " + json.dumps(row, ensure_ascii=False), flush=True)
    return row


async def run_one_scenario(
    engine: AsyncLLM,
    tokenizer,
    scenario: str,
    context_len: int,
    max_tokens: int,
    token_pool_words: int,
    num_followup_requests: int,
    promotion_min_access_count: int,
    post_request_settle_ms: float,
    promotion_mode: str,
    experiment_boundary: str,
    state_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = build_token_pool(tokenizer, f"phase2_{scenario}", token_pool_words)
    if len(pool) < context_len:
        raise ValueError(f"Token pool too small for {scenario}: have={len(pool)}, need={context_len}")
    prompt_tokens = pool[:context_len]
    prefix_id = prefix_id_for(prompt_tokens)
    sm = Phase2StateMachine(prefix_id, scenario)
    rows: list[dict[str, Any]] = []

    reference_row = await emit_request(
        engine,
        scenario,
        prefix_id,
        "full_reference",
        0,
        sm.state,
        sm.state,
        prompt_tokens,
        max_tokens,
        "full",
        None,
        "quality_reference_skip_save",
        skip_save=True,
    )
    rows.append(reference_row)
    reference_output_token_ids = reference_row["output_token_ids"]
    if post_request_settle_ms > 0:
        await asyncio.sleep(post_request_settle_ms / 1000.0)

    if scenario == "oracle_full_ready":
        prime_fidelity = "full"
        prime_reason = "oracle_prime_full_namespace"
    else:
        prime_fidelity = "base"
        prime_reason = "prime_base_namespace"

    before = sm.state
    prime_row = await emit_request(
        engine,
        scenario,
        prefix_id,
        "prime",
        0,
        before,
        PrefixState.FULL_READY if scenario == "oracle_full_ready" else PrefixState.BASE_READY,
        prompt_tokens,
        max_tokens,
        prime_fidelity,
        reference_output_token_ids,
        prime_reason,
    )
    rows.append(prime_row)
    if scenario == "oracle_full_ready":
        sm.transition(PrefixState.FULL_READY, "MISS_TO_FULL_READY", reason="oracle_prime_full_namespace")
    else:
        sm.mark_base_ready(reason="base_prime_complete")
    if post_request_settle_ms > 0:
        await asyncio.sleep(post_request_settle_ms / 1000.0)

    promoted = False
    for idx in range(1, num_followup_requests + 1):
        fidelity, reason = sm.choose_auto_fidelity(idx, state_source=state_source)

        state_before = sm.state
        state_after = sm.state
        row = await emit_request(
            engine,
            scenario,
            prefix_id,
            "followup",
            idx,
            state_before,
            state_after,
            prompt_tokens,
            max_tokens,
            fidelity,
            reference_output_token_ids,
            reason,
            use_core_auto=True,
            state_source=state_source,
            enable_internal_promotion=(scenario == "promotion")
            if state_source == "internal"
            else None,
        )
        rows.append(row)

        should_promote = False
        if fidelity == "base":
            should_promote = sm.observe_base_hit(
                promotion_min_access_count,
                enable_promotion=(scenario == "promotion"),
            )
        else:
            sm.observe_full_hit(promoted=(scenario == "promotion"))

        if post_request_settle_ms > 0:
            await asyncio.sleep(post_request_settle_ms / 1000.0)

        if should_promote:
            promotion_start = time.perf_counter()
            try:
                promo_row = await emit_request(
                    engine,
                    scenario,
                    prefix_id,
                    "promotion_materialize_full",
                    idx,
                    PrefixState.PROMOTING,
                    PrefixState.FULL_READY,
                    prompt_tokens,
                    max_tokens,
                    "full",
                    reference_output_token_ids,
                    "full_disk_retrieve_writeback_after_dual_write"
                    if promotion_mode == "request_driven_full_disk_retrieve_writeback"
                    else "full_materialization_proxy_after_base_hit",
                    state_source=state_source,
                )
                rows.append(promo_row)
                latency_ms = round((time.perf_counter() - promotion_start) * 1000.0, 3)
                sm.stats.promotion_success += 1
                sm.mark_full_ready(
                    "full_disk_retrieve_writeback_success"
                    if promotion_mode == "request_driven_full_disk_retrieve_writeback"
                    else "full_materialization_proxy_success",
                    promotion_latency_ms=latency_ms,
                )
                promoted = True
            except Exception as exc:  # pragma: no cover - experiment safety path
                sm.promotion_failed(type(exc).__name__ + ": " + str(exc))
                raise
            if post_request_settle_ms > 0:
                await asyncio.sleep(post_request_settle_ms / 1000.0)

    followups = [row for row in rows if row.get("request_kind") == "followup"]
    subsequent = [row for row in followups if row.get("sequence_index", 0) >= 2]
    summary = sm.summary_dict()
    summary.update(
        {
            "context_len": context_len,
            "max_tokens": max_tokens,
            "num_followup_requests": num_followup_requests,
            "first_followup_ttft_ms": followups[0].get("ttft_ms") if followups else None,
            "mean_followup_ttft_ms": round(statistics.mean([r["ttft_ms"] for r in followups if r.get("ttft_ms") is not None]), 3) if followups else None,
            "mean_subsequent_ttft_ms": round(statistics.mean([r["ttft_ms"] for r in subsequent if r.get("ttft_ms") is not None]), 3) if subsequent else None,
            "mean_followup_match_rate": round(statistics.mean([r["output_match_rate_vs_full_ref"] for r in followups if r.get("output_match_rate_vs_full_ref") is not None]), 6) if followups else None,
            "promotion_mode": promotion_mode,
            "experiment_boundary": experiment_boundary,
            "state_source": state_source,
        }
    )
    print("PHASE2_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return rows, summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--run-scenarios", nargs="+", default=["no_promotion", "promotion", "oracle_full_ready"], choices=["no_promotion", "promotion", "oracle_full_ready"])
    parser.add_argument("--num-followup-requests", type=int, default=3)
    parser.add_argument("--promotion-min-access-count", type=int, default=1)
    parser.add_argument("--base-codec", default="int8", choices=["fake", "int8"])
    parser.add_argument("--use-layerwise", action="store_true")
    parser.add_argument("--enable-async-loading", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--token-pool-words", type=int, default=60000)
    parser.add_argument("--post-request-settle-ms", type=float, default=250.0)
    parser.add_argument("--max-local-cpu-size", type=float, default=8.0)
    parser.add_argument("--dual-write-full-to-disk", action="store_true")
    parser.add_argument("--local-disk", default="")
    parser.add_argument("--max-local-disk-size", type=float, default=0.0)
    parser.add_argument("--store-base-target", default="local_cpu")
    parser.add_argument("--store-full-target", default="local_disk")
    parser.add_argument("--state-source", choices=["harness", "internal"], default="harness")
    parser.add_argument("--use-internal-state-store", action="store_true")
    args = parser.parse_args()
    state_source = "internal" if args.use_internal_state_store else args.state_source

    os.environ["LMCACHE_ENABLE_FIDELITY_CACHE"] = "True"
    os.environ["LMCACHE_DEFAULT_FIDELITY"] = "auto"
    os.environ["LMCACHE_BASE_CODEC"] = args.base_codec
    os.environ["LMCACHE_USE_LAYERWISE"] = "True" if args.use_layerwise else "False"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "True" if args.enable_async_loading else "False"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.max_local_cpu_size)
    os.environ["LMCACHE_ENABLE_FIDELITY_INTERNAL_STATE"] = (
        "True" if state_source == "internal" else "False"
    )
    os.environ["LMCACHE_FIDELITY_INTERNAL_STATE_PROMOTION_MIN_ACCESS_COUNT"] = str(
        args.promotion_min_access_count
    )
    os.environ["LMCACHE_FIDELITY_INTERNAL_STATE_ENABLE_PROMOTION"] = "False"
    if args.dual_write_full_to_disk:
        os.environ["LMCACHE_STORE_BASE_VARIANT"] = "True"
        os.environ["LMCACHE_STORE_FULL_VARIANT"] = "True"
        os.environ["LMCACHE_STORE_BASE_TARGET"] = args.store_base_target
        os.environ["LMCACHE_STORE_FULL_TARGET"] = args.store_full_target
        if args.local_disk:
            os.environ["LMCACHE_LOCAL_DISK"] = args.local_disk
        if args.max_local_disk_size > 0:
            os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(args.max_local_disk_size)

    max_model_len = args.max_model_len or (args.context_len + args.max_tokens + 512)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**9)

    python_fallback = bool(getattr(lmc_ops, "PYTHON_FALLBACK", False))
    c_ops_available = not python_fallback
    promotion_mode = current_promotion_mode(args.use_layerwise)
    experiment_boundary = current_experiment_boundary(args.use_layerwise)

    config_row = {
        "event": "config",
        "model": args.model,
        "context_len": args.context_len,
        "run_scenarios": args.run_scenarios,
        "num_followup_requests": args.num_followup_requests,
        "promotion_min_access_count": args.promotion_min_access_count,
        "base_codec": args.base_codec,
        "use_layerwise": args.use_layerwise,
        "enable_async_loading": args.enable_async_loading,
        "max_tokens": args.max_tokens,
        "max_model_len": max_model_len,
        "python_fallback": python_fallback,
        "c_ops_available": c_ops_available,
        "dual_write_full_to_disk_arg": args.dual_write_full_to_disk,
        "store_base_variant": env_bool("LMCACHE_STORE_BASE_VARIANT"),
        "store_full_variant": env_bool("LMCACHE_STORE_FULL_VARIANT", True),
        "cleanup_base_on_full_store": env_bool("LMCACHE_CLEANUP_BASE_ON_FULL_STORE"),
        "store_base_target": os.environ.get("LMCACHE_STORE_BASE_TARGET", "local_cpu"),
        "store_full_target": os.environ.get("LMCACHE_STORE_FULL_TARGET", "local_cpu"),
        "local_disk": os.environ.get("LMCACHE_LOCAL_DISK"),
        "max_local_disk_size": os.environ.get("LMCACHE_MAX_LOCAL_DISK_SIZE"),
        "auto_policy": "core_auto_state_store_v0",
        "state_source": state_source,
        "enable_fidelity_internal_state": env_bool("LMCACHE_ENABLE_FIDELITY_INTERNAL_STATE"),
        "promotion_mode": promotion_mode,
        "boundary": experiment_boundary,
    }
    print("PHASE2_CONFIG " + json.dumps(config_row, ensure_ascii=False), flush=True)

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        disable_log_stats=True,
        kv_transfer_config=KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both"),
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    summaries: list[dict[str, Any]] = []
    try:
        for scenario in args.run_scenarios:
            _, summary = await run_one_scenario(
                engine,
                tokenizer,
                scenario,
                args.context_len,
                args.max_tokens,
                args.token_pool_words,
                args.num_followup_requests,
                args.promotion_min_access_count,
                args.post_request_settle_ms,
                promotion_mode,
                experiment_boundary,
                state_source,
            )
            summaries.append(summary)
    finally:
        engine.shutdown(timeout=0)

    print("PHASE2_ALL_SUMMARIES " + json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
