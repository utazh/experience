#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lmcache.c_ops as lmc_ops
from transformers import AutoTokenizer
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from benchmarks.quality_aware_kv.phase2.run_phase2_hot_prefix import (
    Phase2StateMachine,
    PrefixState,
    current_experiment_boundary,
    current_promotion_mode,
    emit_request,
    env_bool,
    prefix_id_for,
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def fit_prompt_tokens(tokenizer, prefix_text: str, suffix_text: str, context_len: int) -> list[int]:
    suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)
    if len(suffix_tokens) >= context_len:
        return suffix_tokens[-context_len:]
    prefix_budget = context_len - len(suffix_tokens)
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    return (prefix_tokens[:prefix_budget] + suffix_tokens)[:context_len]


def load_hotpotqa_prefixes(tokenizer, path: Path, context_len: int, count: int) -> list[dict[str, Any]]:
    prefixes = []
    for row_index, row in enumerate(load_jsonl(path)):
        question = normalize_text(row.get("input") or row.get("question"))
        context = normalize_text(row.get("context"))
        prefix_text = (
            "You are given a long HotpotQA context. Answer the question using only the context.\n\n"
            f"Context:\n{context}\n\n"
        )
        suffix_text = f"Question:\n{question}\n\nAnswer:"
        tokens = fit_prompt_tokens(tokenizer, prefix_text, suffix_text, context_len)
        if len(tokens) < context_len:
            continue
        prefixes.append({
            "dataset": "hotpotqa",
            "dataset_path": str(path),
            "row_index": row_index,
            "record_id": row.get("_id"),
            "question": question[:240],
            "answers": row.get("answers"),
            "source_length": row.get("length"),
            "prompt_tokens": tokens,
            "construction": "HotpotQA real context/question; context is head-truncated if needed while preserving the question suffix.",
        })
        if len(prefixes) >= count:
            break
    if len(prefixes) < count:
        raise ValueError(f"Need {count} HotpotQA rows with enough tokens, got {len(prefixes)} from {path}")
    return prefixes


def format_sharegpt_conversation(row: dict[str, Any]) -> str:
    parts = []
    for turn in row.get("conversations", []):
        role = normalize_text(turn.get("from") or turn.get("role") or "unknown")
        value = normalize_text(turn.get("value") or turn.get("content"))
        label = "Human" if role.lower() in {"human", "user"} else "Assistant" if role.lower() in {"gpt", "assistant"} else role
        parts.append(f"{label}: {value}")
    return "\n".join(parts)


def load_sharegpt_prefixes(tokenizer, path: Path, context_len: int, count: int) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"ShareGPT file is missing or empty: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    prefixes = []
    cursor = 0
    for prefix_idx in range(count):
        chunks = ["The following is a real ShareGPT multi-turn conversation transcript. Continue as the assistant.\n"]
        start_cursor = cursor
        source_ids = []
        while cursor < len(data):
            row = data[cursor]
            cursor += 1
            text = format_sharegpt_conversation(row)
            if not text.strip():
                continue
            chunks.append(f"\n[Conversation {cursor}]\n{text}\n")
            source_ids.append(row.get("id") or row.get("_id") or cursor - 1)
            toks = tokenizer.encode("".join(chunks), add_special_tokens=False)
            if len(toks) >= context_len:
                tokens = toks[:context_len]
                prefixes.append({
                    "dataset": "sharegpt",
                    "dataset_path": str(path),
                    "row_index": start_cursor,
                    "record_id": source_ids[0] if source_ids else None,
                    "source_ids": source_ids[:8],
                    "source_conversation_count": len(source_ids),
                    "prompt_tokens": tokens,
                    "construction": "Concatenated real ShareGPT conversation transcripts, head-truncated to the target context length.",
                })
                break
        if len(prefixes) <= prefix_idx:
            break
    if len(prefixes) < count:
        raise ValueError(f"Need {count} ShareGPT prefixes with >= {context_len} tokens, got {len(prefixes)} from {path}")
    return prefixes


def load_prefixes(dataset: str, tokenizer, path: Path, context_len: int, count: int) -> list[dict[str, Any]]:
    if dataset == "hotpotqa":
        return load_hotpotqa_prefixes(tokenizer, path, context_len, count)
    if dataset == "sharegpt":
        return load_sharegpt_prefixes(tokenizer, path, context_len, count)
    raise ValueError(f"Unsupported dataset: {dataset}")


async def run_real_prefix_scenario(
    engine: AsyncLLM,
    prefix_meta: dict[str, Any],
    scenario: str,
    max_tokens: int,
    num_followup_requests: int,
    promotion_min_access_count: int,
    post_request_settle_ms: float,
    promotion_mode: str,
    experiment_boundary: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_tokens = prefix_meta["prompt_tokens"]
    prefix_id = prefix_id_for(prompt_tokens)
    sm = Phase2StateMachine(prefix_id, scenario)
    rows: list[dict[str, Any]] = []

    def annotate(row: dict[str, Any]) -> dict[str, Any]:
        row.update({k: v for k, v in prefix_meta.items() if k != "prompt_tokens"})
        row["dataset"] = prefix_meta["dataset"]
        row["state_source"] = "internal"
        row["real_dataset"] = True
        return row

    reference_row = await emit_request(
        engine, scenario, prefix_id, "full_reference", 0, sm.state, sm.state,
        prompt_tokens, max_tokens, "full", None, "quality_reference_skip_save", skip_save=True,
        state_source="internal",
    )
    rows.append(annotate(reference_row))
    reference_output_token_ids = reference_row["output_token_ids"]
    if post_request_settle_ms > 0:
        await asyncio.sleep(post_request_settle_ms / 1000.0)

    prime_fidelity = "full" if scenario == "oracle_full_ready" else "base"
    prime_after = PrefixState.FULL_READY if scenario == "oracle_full_ready" else PrefixState.BASE_READY
    prime_reason = "oracle_prime_full_namespace" if scenario == "oracle_full_ready" else "prime_base_namespace"
    prime_row = await emit_request(
        engine, scenario, prefix_id, "prime", 0, sm.state, prime_after,
        prompt_tokens, max_tokens, prime_fidelity, reference_output_token_ids, prime_reason,
        state_source="internal",
    )
    rows.append(annotate(prime_row))
    if scenario == "oracle_full_ready":
        sm.transition(PrefixState.FULL_READY, "MISS_TO_FULL_READY", reason="oracle_prime_full_namespace")
    else:
        sm.mark_base_ready(reason="base_prime_complete")
    if post_request_settle_ms > 0:
        await asyncio.sleep(post_request_settle_ms / 1000.0)

    promoted = False
    for idx in range(1, num_followup_requests + 1):
        fidelity, reason = sm.choose_auto_fidelity(idx, state_source="internal")
        state_before = sm.state
        row = await emit_request(
            engine, scenario, prefix_id, "followup", idx, state_before, state_before,
            prompt_tokens, max_tokens, fidelity, reference_output_token_ids, reason,
            use_core_auto=True, state_source="internal",
            enable_internal_promotion=(scenario == "promotion"),
        )
        rows.append(annotate(row))
        should_promote = False
        if fidelity == "base":
            should_promote = sm.observe_base_hit(promotion_min_access_count, enable_promotion=(scenario == "promotion"))
        else:
            sm.observe_full_hit(promoted=(scenario == "promotion"))
        if post_request_settle_ms > 0:
            await asyncio.sleep(post_request_settle_ms / 1000.0)
        if should_promote:
            promotion_start = time.perf_counter()
            promo_row = await emit_request(
                engine, scenario, prefix_id, "promotion_materialize_full", idx,
                PrefixState.PROMOTING, PrefixState.FULL_READY, prompt_tokens, max_tokens,
                "full", reference_output_token_ids,
                "full_disk_retrieve_writeback_after_dual_write" if promotion_mode == "request_driven_full_disk_retrieve_writeback" else "full_materialization_proxy_after_base_hit",
                state_source="internal",
            )
            rows.append(annotate(promo_row))
            latency_ms = round((time.perf_counter() - promotion_start) * 1000.0, 3)
            sm.stats.promotion_success += 1
            sm.mark_full_ready("full_disk_retrieve_writeback_success", promotion_latency_ms=latency_ms)
            promoted = True
            if post_request_settle_ms > 0:
                await asyncio.sleep(post_request_settle_ms / 1000.0)

    followups = [row for row in rows if row.get("request_kind") == "followup"]
    subsequent = [row for row in followups if row.get("sequence_index", 0) >= 2]
    summary = sm.summary_dict()
    summary.update({k: v for k, v in prefix_meta.items() if k != "prompt_tokens"})
    summary.update({
        "context_len": len(prompt_tokens),
        "max_tokens": max_tokens,
        "num_followup_requests": num_followup_requests,
        "first_followup_ttft_ms": followups[0].get("ttft_ms") if followups else None,
        "mean_followup_ttft_ms": round(statistics.mean([r["ttft_ms"] for r in followups if r.get("ttft_ms") is not None]), 3) if followups else None,
        "mean_subsequent_ttft_ms": round(statistics.mean([r["ttft_ms"] for r in subsequent if r.get("ttft_ms") is not None]), 3) if subsequent else None,
        "mean_followup_match_rate": round(statistics.mean([r["output_match_rate_vs_full_ref"] for r in followups if r.get("output_match_rate_vs_full_ref") is not None]), 6) if followups else None,
        "promotion_mode": promotion_mode,
        "experiment_boundary": experiment_boundary,
        "state_source": "internal",
        "real_dataset": True,
    })
    print("PHASE2_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return rows, summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", choices=["hotpotqa", "sharegpt"], required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--context-len", type=int, default=16384)
    parser.add_argument("--num-prefixes", type=int, default=1)
    parser.add_argument("--run-scenarios", nargs="+", default=["no_promotion", "promotion", "oracle_full_ready"], choices=["no_promotion", "promotion", "oracle_full_ready"])
    parser.add_argument("--num-followup-requests", type=int, default=6)
    parser.add_argument("--promotion-min-access-count", type=int, default=1)
    parser.add_argument("--base-codec", default="int8", choices=["fake", "int8"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--post-request-settle-ms", type=float, default=250.0)
    parser.add_argument("--max-local-cpu-size", type=float, default=10.0)
    parser.add_argument("--local-disk", required=True)
    parser.add_argument("--max-local-disk-size", type=float, default=80.0)
    args = parser.parse_args()

    os.environ["LMCACHE_ENABLE_FIDELITY_CACHE"] = "True"
    os.environ["LMCACHE_DEFAULT_FIDELITY"] = "auto"
    os.environ["LMCACHE_ENABLE_FIDELITY_INTERNAL_STATE"] = "True"
    os.environ["LMCACHE_FIDELITY_INTERNAL_STATE_PROMOTION_MIN_ACCESS_COUNT"] = str(args.promotion_min_access_count)
    os.environ["LMCACHE_FIDELITY_INTERNAL_STATE_ENABLE_PROMOTION"] = "False"
    os.environ["LMCACHE_BASE_CODEC"] = args.base_codec
    os.environ["LMCACHE_USE_LAYERWISE"] = "False"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "False"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.max_local_cpu_size)
    os.environ["LMCACHE_STORE_BASE_VARIANT"] = "True"
    os.environ["LMCACHE_STORE_FULL_VARIANT"] = "True"
    os.environ["LMCACHE_STORE_BASE_TARGET"] = "local_cpu"
    os.environ["LMCACHE_STORE_FULL_TARGET"] = "local_disk"
    os.environ["LMCACHE_LOCAL_DISK"] = args.local_disk
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(args.max_local_disk_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**9)
    prefixes = load_prefixes(args.dataset, tokenizer, Path(args.dataset_path), args.context_len, args.num_prefixes)
    max_model_len = args.max_model_len or (args.context_len + args.max_tokens + 512)
    python_fallback = bool(getattr(lmc_ops, "PYTHON_FALLBACK", False))
    promotion_mode = current_promotion_mode(False)
    experiment_boundary = current_experiment_boundary(False)
    config_row = {
        "event": "config",
        "workload": "real_dataset_internal_state_hot_prefix",
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "model": args.model,
        "context_len": args.context_len,
        "num_prefixes": args.num_prefixes,
        "run_scenarios": args.run_scenarios,
        "num_followup_requests": args.num_followup_requests,
        "promotion_min_access_count": args.promotion_min_access_count,
        "base_codec": args.base_codec,
        "max_tokens": args.max_tokens,
        "max_model_len": max_model_len,
        "python_fallback": python_fallback,
        "c_ops_available": not python_fallback,
        "state_source": "internal",
        "enable_fidelity_internal_state": env_bool("LMCACHE_ENABLE_FIDELITY_INTERNAL_STATE"),
        "store_base_variant": env_bool("LMCACHE_STORE_BASE_VARIANT"),
        "store_full_variant": env_bool("LMCACHE_STORE_FULL_VARIANT", True),
        "cleanup_base_on_full_store": env_bool("LMCACHE_CLEANUP_BASE_ON_FULL_STORE"),
        "store_base_target": os.environ.get("LMCACHE_STORE_BASE_TARGET"),
        "store_full_target": os.environ.get("LMCACHE_STORE_FULL_TARGET"),
        "local_disk": args.local_disk,
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
    summaries = []
    try:
        for prefix_idx, prefix_meta in enumerate(prefixes):
            prefix_meta["prefix_index"] = prefix_idx
            for scenario in args.run_scenarios:
                _, summary = await run_real_prefix_scenario(
                    engine, prefix_meta, scenario, args.max_tokens, args.num_followup_requests,
                    args.promotion_min_access_count, args.post_request_settle_ms,
                    promotion_mode, experiment_boundary,
                )
                summaries.append(summary)
    finally:
        engine.shutdown(timeout=0)
    print("PHASE2_ALL_SUMMARIES " + json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
