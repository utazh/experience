#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any

import lmcache.c_ops as lmc_ops
from transformers import AutoTokenizer
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from run_phase2_hot_prefix import (
    Phase2StateMachine,
    PrefixState,
    build_token_pool,
    current_experiment_boundary,
    current_promotion_mode,
    emit_request,
    env_bool,
    prefix_id_for,
)


def configure_env(args: argparse.Namespace) -> None:
    os.environ["LMCACHE_ENABLE_FIDELITY_CACHE"] = "True"
    os.environ["LMCACHE_DEFAULT_FIDELITY"] = "auto"
    os.environ["LMCACHE_BASE_CODEC"] = args.base_codec
    os.environ["LMCACHE_USE_LAYERWISE"] = "True" if args.use_layerwise else "False"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "True" if args.enable_async_loading else "False"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.max_local_cpu_size)
    if args.dual_write_full_to_disk:
        os.environ["LMCACHE_STORE_BASE_VARIANT"] = "True"
        os.environ["LMCACHE_STORE_FULL_VARIANT"] = "True"
        os.environ["LMCACHE_STORE_BASE_TARGET"] = args.store_base_target
        os.environ["LMCACHE_STORE_FULL_TARGET"] = args.store_full_target
        if args.local_disk:
            os.environ["LMCACHE_LOCAL_DISK"] = args.local_disk
        if args.max_local_disk_size > 0:
            os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(args.max_local_disk_size)


async def build_reference(
    engine: AsyncLLM,
    scenario: str,
    prefix_id: str,
    prompt_tokens: list[int],
    max_tokens: int,
) -> list[int]:
    row = await emit_request(
        engine,
        scenario,
        prefix_id,
        "full_reference",
        0,
        PrefixState.MISS,
        PrefixState.MISS,
        prompt_tokens,
        max_tokens,
        "full",
        None,
        "quality_reference_skip_save",
        skip_save=True,
    )
    return row["output_token_ids"]


async def prime_base(
    engine: AsyncLLM,
    sm: Phase2StateMachine,
    scenario: str,
    prefix_id: str,
    prompt_tokens: list[int],
    max_tokens: int,
    reference_output_token_ids: list[int],
    settle_s: float,
) -> dict[str, Any]:
    row = await emit_request(
        engine,
        scenario,
        prefix_id,
        "prime",
        0,
        sm.state,
        PrefixState.BASE_READY,
        prompt_tokens,
        max_tokens,
        "base",
        reference_output_token_ids,
        "prime_base_namespace",
    )
    sm.mark_base_ready("base_prime_complete")
    if settle_s > 0:
        await asyncio.sleep(settle_s)
    return row


async def base_followup_maybe_promote(
    engine: AsyncLLM,
    sm: Phase2StateMachine,
    scenario: str,
    prefix_id: str,
    sequence_index: int,
    prompt_tokens: list[int],
    max_tokens: int,
    reference_output_token_ids: list[int],
    promotion_min_access_count: int,
    promotion_mode: str,
    settle_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fidelity, reason = sm.choose_auto_fidelity(sequence_index)
    row = await emit_request(
        engine,
        scenario,
        prefix_id,
        "followup",
        sequence_index,
        sm.state,
        sm.state,
        prompt_tokens,
        max_tokens,
        fidelity,
        reference_output_token_ids,
        reason,
        use_core_auto=True,
    )
    rows.append(row)
    if fidelity != "base":
        sm.observe_full_hit(promoted=True)
        if settle_s > 0:
            await asyncio.sleep(settle_s)
        return rows
    should_promote = sm.observe_base_hit(promotion_min_access_count, enable_promotion=True)
    if settle_s > 0:
        await asyncio.sleep(settle_s)
    if should_promote:
        promotion_start = time.perf_counter()
        promo_row = await emit_request(
            engine,
            scenario,
            prefix_id,
            "promotion_materialize_full",
            sequence_index,
            PrefixState.PROMOTING,
            PrefixState.FULL_READY,
            prompt_tokens,
            max_tokens,
            "full",
            reference_output_token_ids,
            "full_disk_retrieve_writeback_after_dual_write"
            if promotion_mode == "request_driven_full_disk_retrieve_writeback"
            else "full_materialization_proxy_after_base_hit",
        )
        rows.append(promo_row)
        sm.stats.promotion_success += 1
        sm.mark_full_ready(
            "full_disk_retrieve_writeback_success"
            if promotion_mode == "request_driven_full_disk_retrieve_writeback"
            else "full_materialization_proxy_success",
            promotion_latency_ms=round((time.perf_counter() - promotion_start) * 1000.0, 3),
        )
        if settle_s > 0:
            await asyncio.sleep(settle_s)
    return rows


async def full_followup(
    engine: AsyncLLM,
    sm: Phase2StateMachine,
    scenario: str,
    prefix_id: str,
    sequence_index: int,
    prompt_tokens: list[int],
    max_tokens: int,
    reference_output_token_ids: list[int],
    settle_s: float,
) -> dict[str, Any]:
    row = await emit_request(
        engine,
        scenario,
        prefix_id,
        "followup",
        sequence_index,
        sm.state,
        sm.state,
        prompt_tokens,
        max_tokens,
        "full",
        reference_output_token_ids,
        "full_ready_hit",
        use_core_auto=True,
    )
    sm.observe_full_hit(promoted=True)
    if settle_s > 0:
        await asyncio.sleep(settle_s)
    return row


async def run_mixed(args: argparse.Namespace) -> dict[str, Any]:
    configure_env(args)
    max_model_len = args.max_model_len or (args.context_len + args.max_tokens + 512)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**9)
    promotion_mode = current_promotion_mode(args.use_layerwise)
    experiment_boundary = current_experiment_boundary(args.use_layerwise)

    config = {
        "event": "mixed_config",
        "model": args.model,
        "context_len": args.context_len,
        "hot_followups": args.hot_followups,
        "cold_followups": args.cold_followups,
        "promotion_min_access_count": args.promotion_min_access_count,
        "base_codec": args.base_codec,
        "use_layerwise": args.use_layerwise,
        "max_tokens": args.max_tokens,
        "max_model_len": max_model_len,
        "python_fallback": bool(getattr(lmc_ops, "PYTHON_FALLBACK", False)),
        "c_ops_available": not bool(getattr(lmc_ops, "PYTHON_FALLBACK", False)),
        "dual_write_full_to_disk_arg": args.dual_write_full_to_disk,
        "store_base_variant": env_bool("LMCACHE_STORE_BASE_VARIANT"),
        "store_full_variant": env_bool("LMCACHE_STORE_FULL_VARIANT", True),
        "store_base_target": os.environ.get("LMCACHE_STORE_BASE_TARGET", "local_cpu"),
        "store_full_target": os.environ.get("LMCACHE_STORE_FULL_TARGET", "local_cpu"),
        "local_disk": os.environ.get("LMCACHE_LOCAL_DISK"),
        "max_local_disk_size": os.environ.get("LMCACHE_MAX_LOCAL_DISK_SIZE"),
        "auto_policy": "core_auto_state_store_v0",
        "promotion_mode": promotion_mode,
        "boundary": experiment_boundary,
    }
    print("PHASE2_MIXED_CONFIG " + json.dumps(config, ensure_ascii=False), flush=True)

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
    settle_s = args.post_request_settle_ms / 1000.0
    rows: list[dict[str, Any]] = []

    hot_pool = build_token_pool(tokenizer, "phase2_mixed_hot", args.token_pool_words)
    cold_pool = build_token_pool(tokenizer, "phase2_mixed_cold", args.token_pool_words)
    if len(hot_pool) < args.context_len or len(cold_pool) < args.context_len:
        raise ValueError("Token pool too small for mixed experiment")
    hot_tokens = hot_pool[: args.context_len]
    cold_tokens = cold_pool[: args.context_len]
    hot_id = prefix_id_for(hot_tokens)
    cold_id = prefix_id_for(cold_tokens)
    hot_sm = Phase2StateMachine(hot_id, "mixed_hot")
    cold_sm = Phase2StateMachine(cold_id, "mixed_cold")

    try:
        hot_ref = await build_reference(engine, "mixed_hot", hot_id, hot_tokens, args.max_tokens)
        if settle_s > 0:
            await asyncio.sleep(settle_s)
        cold_ref = await build_reference(engine, "mixed_cold", cold_id, cold_tokens, args.max_tokens)
        if settle_s > 0:
            await asyncio.sleep(settle_s)
        rows.append(await prime_base(engine, hot_sm, "mixed_hot", hot_id, hot_tokens, args.max_tokens, hot_ref, settle_s))
        rows.append(await prime_base(engine, cold_sm, "mixed_cold", cold_id, cold_tokens, args.max_tokens, cold_ref, settle_s))

        sequence_index = 1
        for _ in range(args.cold_followups):
            rows.extend(
                await base_followup_maybe_promote(
                    engine,
                    cold_sm,
                    "mixed_cold",
                    cold_id,
                    sequence_index,
                    cold_tokens,
                    args.max_tokens,
                    cold_ref,
                    args.promotion_min_access_count,
                    promotion_mode,
                    settle_s,
                )
            )
            sequence_index += 1

        for idx in range(1, args.hot_followups + 1):
            if hot_sm.state == PrefixState.FULL_READY:
                rows.append(await full_followup(engine, hot_sm, "mixed_hot", hot_id, idx, hot_tokens, args.max_tokens, hot_ref, settle_s))
            else:
                rows.extend(
                    await base_followup_maybe_promote(
                        engine,
                        hot_sm,
                        "mixed_hot",
                        hot_id,
                        idx,
                        hot_tokens,
                        args.max_tokens,
                        hot_ref,
                        args.promotion_min_access_count,
                        promotion_mode,
                        settle_s,
                    )
                )
    finally:
        engine.shutdown(timeout=0)

    hot_summary = hot_sm.summary_dict()
    cold_summary = cold_sm.summary_dict()
    hot_followup_ttft = [r["ttft_ms"] for r in rows if r.get("scenario") == "mixed_hot" and r.get("request_kind") == "followup" and r.get("ttft_ms") is not None]
    cold_followup_ttft = [r["ttft_ms"] for r in rows if r.get("scenario") == "mixed_cold" and r.get("request_kind") == "followup" and r.get("ttft_ms") is not None]
    summary = {
        "event": "mixed_summary",
        "hot": hot_summary,
        "cold": cold_summary,
        "hot_mean_followup_ttft_ms": round(statistics.mean(hot_followup_ttft), 3) if hot_followup_ttft else None,
        "cold_mean_followup_ttft_ms": round(statistics.mean(cold_followup_ttft), 3) if cold_followup_ttft else None,
        "cold_promotion_enqueued": cold_summary["promotion_enqueued"],
        "cold_promotion_success": cold_summary["promotion_success"],
        "hot_promotion_enqueued": hot_summary["promotion_enqueued"],
        "hot_promotion_success": hot_summary["promotion_success"],
        "promotion_mode": promotion_mode,
        "experiment_boundary": experiment_boundary,
    }
    print("PHASE2_MIXED_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--hot-followups", type=int, default=6)
    parser.add_argument("--cold-followups", type=int, default=1)
    parser.add_argument("--promotion-min-access-count", type=int, default=2)
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
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_mixed(parse_args()))
