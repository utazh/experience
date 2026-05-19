#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lmcache.c_ops as lmc_ops
from transformers import AutoTokenizer
from vllm import TokensPrompt
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from benchmarks.quality_aware_kv.phase2.run_phase2_real_dataset import (
    load_sharegpt_prefixes,
)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def emit(event: str, row: dict[str, Any]) -> None:
    print(f"SCHED_OVERLAP_{event} " + json.dumps(row, ensure_ascii=False), flush=True)


async def run_request(
    engine: AsyncLLM,
    req_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    fidelity: str,
    skip_save: bool,
    role: str,
    pair_id: str | None = None,
) -> dict[str, Any]:
    kv_params: dict[str, Any] = {"lmcache.fidelity": fidelity}
    if skip_save:
        kv_params["lmcache.skip_save"] = True
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        output_kind=RequestOutputKind.DELTA,
        extra_args={"kv_transfer_params": kv_params},
    )
    submit_perf = time.perf_counter()
    submit_wall = now_str()
    emit("SUBMIT", {
        "event": "submit",
        "req_id": req_id,
        "role": role,
        "pair_id": pair_id,
        "prompt_tokens": len(prompt_token_ids),
        "fidelity": fidelity,
        "skip_save": skip_save,
        "submit_wall_ts": submit_wall,
    })
    first_perf = None
    first_wall = None
    final_text = ""
    output_token_ids: list[int] = []
    engine_req_id = req_id
    async for out in engine.generate(
        TokensPrompt(prompt_token_ids=prompt_token_ids),
        params,
        request_id=req_id,
    ):
        if first_perf is None:
            first_perf = time.perf_counter()
            first_wall = now_str()
        engine_req_id = getattr(out, "request_id", engine_req_id)
        if out.outputs:
            sample = out.outputs[0]
            final_text += sample.text
            token_ids = getattr(sample, "token_ids", None)
            if token_ids:
                output_token_ids.extend(list(token_ids))
    end_perf = time.perf_counter()
    row = {
        "event": "request_result",
        "req_id": req_id,
        "engine_req_id": engine_req_id,
        "role": role,
        "pair_id": pair_id,
        "prompt_tokens": len(prompt_token_ids),
        "fidelity": fidelity,
        "skip_save": skip_save,
        "submit_wall_ts": submit_wall,
        "first_token_wall_ts": first_wall,
        "end_wall_ts": now_str(),
        "ttft_ms": None if first_perf is None else round((first_perf - submit_perf) * 1000.0, 3),
        "e2e_ms": round((end_perf - submit_perf) * 1000.0, 3),
        "output_token_count": len(output_token_ids),
        "output_preview": final_text[:200],
        "output_token_ids": output_token_ids,
    }
    emit("REQ", row)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-path", default="/data1/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--context-len", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--miss-delay-ms", type=float, default=20.0)
    parser.add_argument("--post-request-settle-ms", type=float, default=150.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-local-cpu-size", type=float, default=20.0)
    parser.add_argument("--local-disk", required=True)
    parser.add_argument("--max-local-disk-size", type=float, default=80.0)
    args = parser.parse_args()

    os.environ["LMCACHE_ENABLE_FIDELITY_CACHE"] = "True"
    os.environ["LMCACHE_DEFAULT_FIDELITY"] = "full"
    os.environ["LMCACHE_ENABLE_FIDELITY_INTERNAL_STATE"] = "False"
    os.environ["LMCACHE_BASE_CODEC"] = "int8"
    os.environ["LMCACHE_USE_LAYERWISE"] = "False"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "False"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.max_local_cpu_size)
    os.environ["LMCACHE_STORE_BASE_VARIANT"] = "False"
    os.environ["LMCACHE_STORE_FULL_VARIANT"] = "True"
    os.environ["LMCACHE_STORE_FULL_TARGET"] = "local_disk"
    os.environ["LMCACHE_LOCAL_DISK"] = args.local_disk
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(args.max_local_disk_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**9)
    prefixes = load_sharegpt_prefixes(tokenizer, Path(args.dataset_path), args.context_len, 3)
    cached_tokens = prefixes[0]["prompt_tokens"]
    miss_baseline_tokens = prefixes[1]["prompt_tokens"]
    miss_pair_tokens = prefixes[2]["prompt_tokens"]
    max_model_len = args.max_model_len or (args.context_len + args.max_tokens + 512)
    python_fallback = bool(getattr(lmc_ops, "PYTHON_FALLBACK", False))
    config = {
        "event": "config",
        "workload": "scheduler_overlap_opportunity",
        "model": args.model,
        "dataset_path": args.dataset_path,
        "context_len": args.context_len,
        "max_tokens": args.max_tokens,
        "miss_delay_ms": args.miss_delay_ms,
        "max_model_len": max_model_len,
        "python_fallback": python_fallback,
        "c_ops_available": not python_fallback,
        "store_full_target": os.environ.get("LMCACHE_STORE_FULL_TARGET"),
        "local_disk": args.local_disk,
        "boundary": "Diagnostic only: does not modify scheduler. Full-hit request is submitted first; a MISS request is submitted shortly after to test whether it waits while non-layerwise LMCache retrieve blocks the step.",
    }
    emit("CONFIG", config)

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
    rows: list[dict[str, Any]] = []
    try:
        prime = await run_request(
            engine,
            "overlap_prime_full_disk",
            cached_tokens,
            args.max_tokens,
            "full",
            False,
            "prime_full_to_disk",
        )
        rows.append(prime)
        if args.post_request_settle_ms > 0:
            await asyncio.sleep(args.post_request_settle_ms / 1000.0)

        miss_baseline = await run_request(
            engine,
            "overlap_miss_baseline_skip_save",
            miss_baseline_tokens,
            args.max_tokens,
            "full",
            True,
            "miss_baseline",
        )
        rows.append(miss_baseline)
        if args.post_request_settle_ms > 0:
            await asyncio.sleep(args.post_request_settle_ms / 1000.0)

        pair_id = "pair_full_hit_then_miss"
        emit("PAIR", {
            "event": "pair_start",
            "pair_id": pair_id,
            "full_hit_req_id": "overlap_pair_full_hit_skip_save",
            "miss_req_id": "overlap_pair_miss_skip_save",
            "miss_delay_ms": args.miss_delay_ms,
            "start_wall_ts": now_str(),
        })
        full_task = asyncio.create_task(run_request(
            engine,
            "overlap_pair_full_hit_skip_save",
            cached_tokens,
            args.max_tokens,
            "full",
            True,
            "pair_full_hit",
            pair_id,
        ))
        if args.miss_delay_ms > 0:
            await asyncio.sleep(args.miss_delay_ms / 1000.0)
        miss_task = asyncio.create_task(run_request(
            engine,
            "overlap_pair_miss_skip_save",
            miss_pair_tokens,
            args.max_tokens,
            "full",
            True,
            "pair_miss",
            pair_id,
        ))
        pair_rows = await asyncio.gather(full_task, miss_task)
        rows.extend(pair_rows)
        emit("SUMMARY", {
            "event": "summary",
            "rows": rows,
            "miss_baseline_ttft_ms": miss_baseline.get("ttft_ms"),
            "pair_full_hit_ttft_ms": pair_rows[0].get("ttft_ms"),
            "pair_miss_ttft_ms": pair_rows[1].get("ttft_ms"),
            "pair_miss_excess_vs_baseline_ms": None
            if miss_baseline.get("ttft_ms") is None or pair_rows[1].get("ttft_ms") is None
            else round(pair_rows[1]["ttft_ms"] - miss_baseline["ttft_ms"], 3),
        })
    finally:
        engine.shutdown(timeout=0)


if __name__ == "__main__":
    asyncio.run(main())
