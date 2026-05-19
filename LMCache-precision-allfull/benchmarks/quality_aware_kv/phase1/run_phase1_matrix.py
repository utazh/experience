
#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import time
from datetime import datetime

import lmcache.c_ops as lmc_ops
from transformers import AutoTokenizer
from vllm import TokensPrompt
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def build_token_pool(tokenizer, prefix: str, n_words: int) -> list[int]:
    text = " ".join(f"{prefix}_{i}" for i in range(n_words))
    return tokenizer.encode(text, add_special_tokens=False)


def aligned(tokens: int, chunk_size: int = 256) -> int:
    return (tokens // chunk_size) * chunk_size


def build_spans(context_lens: list[int], gap_tokens: int) -> dict[int, tuple[int, int]]:
    spans = {}
    cursor = 0
    for ctx_len in context_lens:
        spans[ctx_len] = (cursor, cursor + ctx_len + gap_tokens)
        cursor += (2 * ctx_len) + (2 * gap_tokens)
    return spans


async def run_request(engine: AsyncLLM, req_id: str, prompt_token_ids: list[int], max_tokens: int, fidelity: str | None):
    extra_args = None
    if fidelity is not None:
        extra_args = {"kv_transfer_params": {"lmcache.fidelity": fidelity}}
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        output_kind=RequestOutputKind.DELTA,
        extra_args=extra_args,
    )
    start = time.perf_counter()
    start_wall_ts = now_str()
    first = None
    first_wall_ts = None
    final_text = ""
    output_token_ids = []
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
    end_wall_ts = now_str()
    return {
        "req_id": req_id,
        "engine_req_id": engine_req_id,
        "start_wall_ts": start_wall_ts,
        "first_token_wall_ts": first_wall_ts,
        "end_wall_ts": end_wall_ts,
        "ttft_ms": None if first is None else round((first - start) * 1000.0, 3),
        "e2e_ms": round((end - start) * 1000.0, 3),
        "output_preview": final_text[:120],
        "output_token_ids": output_token_ids,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-lens", nargs="+", type=int, default=[16384, 32512])
    parser.add_argument("--reuse-buckets", nargs="+", default=["medium", "high"])
    parser.add_argument("--fidelity", default="full", choices=["full", "base", "auto"])
    parser.add_argument("--base-codec", default="int8", choices=["fake", "int8"])
    parser.add_argument("--use-layerwise", action="store_true")
    parser.add_argument("--enable-async-loading", action="store_true")
    parser.add_argument("--gap-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=33000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--token-pool-words", type=int, default=120000)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--post-request-settle-ms", type=float, default=250.0)
    args = parser.parse_args()

    os.environ["LMCACHE_ENABLE_FIDELITY_CACHE"] = "True"
    os.environ["LMCACHE_DEFAULT_FIDELITY"] = args.fidelity
    os.environ["LMCACHE_BASE_CODEC"] = args.base_codec
    os.environ["LMCACHE_USE_LAYERWISE"] = "True" if args.use_layerwise else "False"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "True" if args.enable_async_loading else "False"

    context_lens = sorted(set(args.context_lens))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**9)
    base_pool = build_token_pool(tokenizer, "phase1base", args.token_pool_words)
    alt_pool = build_token_pool(tokenizer, "phase1alt", args.token_pool_words)
    spans = build_spans(context_lens, args.gap_tokens)

    max_needed = 0
    for ctx_len, (base_offset, alt_offset) in spans.items():
        max_needed = max(max_needed, base_offset + ctx_len, alt_offset + ctx_len)
    if len(base_pool) < max_needed or len(alt_pool) < max_needed:
        raise ValueError(
            f"Token pool is too small for requested context lengths. need={max_needed}, "
            f"base_pool={len(base_pool)}, alt_pool={len(alt_pool)}"
        )

    reuse_defs = {
        "low": lambda ctx: 0,
        "medium": lambda ctx: aligned(ctx // 2),
        "high": lambda ctx: ctx,
    }

    python_fallback = bool(getattr(lmc_ops, "PYTHON_FALLBACK", False))
    c_ops_available = not python_fallback
    print(
        "PHASE1_CONFIG " + json.dumps(
            {
                "event": "config",
                "model": args.model,
                "context_lens": context_lens,
                "reuse_buckets": args.reuse_buckets,
                "fidelity": args.fidelity,
                "base_codec": args.base_codec,
                "use_layerwise": args.use_layerwise,
                "enable_async_loading": args.enable_async_loading,
                "max_tokens": args.max_tokens,
                "python_fallback": python_fallback,
                "c_ops_available": c_ops_available,
            }, ensure_ascii=False
        ),
        flush=True,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        disable_log_stats=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        ),
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        for ctx_len in context_lens:
            base_offset, alt_offset = spans[ctx_len]
            warm_tokens = base_pool[base_offset : base_offset + ctx_len]
            warm_req_id = f"warm_{args.fidelity}_ctx{ctx_len}"
            warm_result = await run_request(engine, warm_req_id, warm_tokens, args.max_tokens, args.fidelity)
            warm_row = {
                "event": "request_result",
                "request_kind": "warm",
                "req_id": warm_req_id,
                "model": args.model,
                "context_len": ctx_len,
                "reuse_bucket": "warm",
                "planned_overlap_tokens": 0,
                "prompt_tokens": len(warm_tokens),
                "requested_fidelity": args.fidelity,
                "base_codec": args.base_codec,
                "use_layerwise": args.use_layerwise,
                "enable_async_loading": args.enable_async_loading,
                "python_fallback": python_fallback,
                "c_ops_available": c_ops_available,
            }
            warm_row.update(warm_result)
            print("PHASE1_REQ " + json.dumps(warm_row, ensure_ascii=False), flush=True)
            if args.post_request_settle_ms > 0:
                await asyncio.sleep(args.post_request_settle_ms / 1000.0)

            for bucket in args.reuse_buckets:
                overlap = reuse_defs[bucket](ctx_len)
                suffix_len = ctx_len - overlap
                query_tokens = base_pool[base_offset : base_offset + overlap] + alt_pool[alt_offset : alt_offset + suffix_len]
                req_id = f"{args.fidelity}_ctx{ctx_len}_{bucket}"
                result = await run_request(engine, req_id, query_tokens, args.max_tokens, args.fidelity)
                row = {
                    "event": "request_result",
                    "request_kind": "measure",
                    "req_id": req_id,
                    "model": args.model,
                    "context_len": ctx_len,
                    "reuse_bucket": bucket,
                    "planned_overlap_tokens": overlap,
                    "prompt_tokens": len(query_tokens),
                    "requested_fidelity": args.fidelity,
                    "base_codec": args.base_codec,
                    "use_layerwise": args.use_layerwise,
                    "enable_async_loading": args.enable_async_loading,
                    "python_fallback": python_fallback,
                    "c_ops_available": c_ops_available,
                }
                row.update(result)
                print("PHASE1_REQ " + json.dumps(row, ensure_ascii=False), flush=True)
                if args.post_request_settle_ms > 0:
                    await asyncio.sleep(args.post_request_settle_ms / 1000.0)
    finally:
        engine.shutdown(timeout=0)


if __name__ == "__main__":
    asyncio.run(main())
