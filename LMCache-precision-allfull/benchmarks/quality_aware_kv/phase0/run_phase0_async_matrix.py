#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
from datetime import datetime

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


async def run_request(engine: AsyncLLM, req_id: str, prompt_token_ids: list[int]):
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        output_kind=RequestOutputKind.DELTA,
    )
    start = time.perf_counter()
    start_wall_ts = now_str()
    first = None
    first_wall_ts = None
    final_text = ""
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
            final_text = out.outputs[0].text
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
        "output_preview": final_text[:80],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-lens", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--gap-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--token-pool-words", type=int, default=60000)
    args = parser.parse_args()

    context_lens = sorted(set(args.context_lens))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**9)
    base_pool = build_token_pool(tokenizer, "phase0base", args.token_pool_words)
    alt_pool = build_token_pool(tokenizer, "phase0alt", args.token_pool_words)
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
            warm_req_id = f"warm_ctx{ctx_len}"
            warm_result = await run_request(engine, warm_req_id, warm_tokens)
            warm_row = {
                "event": "request_result",
                "request_kind": "warm",
                "req_id": warm_req_id,
                "model": args.model,
                "context_len": ctx_len,
                "reuse_bucket": "warm",
                "planned_overlap_tokens": 0,
                "prompt_tokens": len(warm_tokens),
                "use_layerwise": False,
                "enable_async_loading": False,
                "python_fallback": True,
                "c_ops_available": False,
            }
            warm_row.update(warm_result)
            print("PHASE0_REQ " + json.dumps(warm_row, ensure_ascii=False), flush=True)

            for bucket, fn in reuse_defs.items():
                overlap = fn(ctx_len)
                suffix_len = ctx_len - overlap
                query_tokens = (
                    base_pool[base_offset : base_offset + overlap]
                    + alt_pool[alt_offset : alt_offset + suffix_len]
                )
                req_id = f"ctx{ctx_len}_{bucket}"
                result = await run_request(engine, req_id, query_tokens)
                row = {
                    "event": "request_result",
                    "request_kind": "measure",
                    "req_id": req_id,
                    "model": args.model,
                    "context_len": ctx_len,
                    "reuse_bucket": bucket,
                    "planned_overlap_tokens": overlap,
                    "prompt_tokens": len(query_tokens),
                    "use_layerwise": False,
                    "enable_async_loading": False,
                    "python_fallback": True,
                    "c_ops_available": False,
                }
                row.update(result)
                print("PHASE0_REQ " + json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        engine.shutdown(timeout=0)


if __name__ == "__main__":
    asyncio.run(main())
