#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import lmcache.c_ops as lmc_ops
from transformers import AutoTokenizer
from vllm import TokensPrompt
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prompt_text(row: dict) -> str:
    context = row.get("context", "")
    question = row.get("input") or row.get("question") or ""
    return (
        "You are given a long document from NarrativeQA. Answer the question using the document.\n\n"
        f"Document:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def build_real_medium_tokens(tokenizer, dataset_path: Path, context_len: int, overlap_tokens: int):
    candidates = []
    for idx, row in enumerate(load_jsonl(dataset_path)):
        toks = tokenizer.encode(prompt_text(row), add_special_tokens=False)
        if len(toks) >= context_len:
            candidates.append((idx, row, toks))
        if len(candidates) >= 2:
            break
    if len(candidates) < 2:
        raise ValueError(f"Need at least two NarrativeQA rows with >= {context_len} tokens")
    warm_idx, warm_row, warm_full = candidates[0]
    alt_idx, alt_row, alt_full = candidates[1]
    warm_tokens = warm_full[:context_len]
    query_tokens = warm_full[:overlap_tokens] + alt_full[overlap_tokens:context_len]
    assert len(warm_tokens) == context_len
    assert len(query_tokens) == context_len
    meta = {
        "dataset": "LongBench/narrativeqa",
        "dataset_path": str(dataset_path),
        "context_len": context_len,
        "reuse_bucket": "medium",
        "overlap_tokens": overlap_tokens,
        "warm_row_index": warm_idx,
        "query_alt_row_index": alt_idx,
        "warm_id": warm_row.get("_id"),
        "query_alt_id": alt_row.get("_id"),
        "warm_question": warm_row.get("input"),
        "query_alt_question": alt_row.get("input"),
        "warm_answers": warm_row.get("answers"),
        "query_alt_answers": alt_row.get("answers"),
        "warm_source_prompt_tokens": len(warm_full),
        "query_alt_source_prompt_tokens": len(alt_full),
        "construction": "query_tokens = warm_tokens[:overlap] + alt_tokens[overlap:context_len]; real LongBench tokens with controlled prefix reuse",
    }
    return warm_tokens, query_tokens, meta


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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", default="/data1/datasets/LongBench/data/narrativeqa.jsonl")
    parser.add_argument("--context-len", type=int, default=16384)
    parser.add_argument("--overlap-tokens", type=int, default=8192)
    parser.add_argument("--fidelity", default="full", choices=["full", "base", "auto"])
    parser.add_argument("--base-codec", default="int8", choices=["fake", "int8"])
    parser.add_argument("--use-layerwise", action="store_true")
    parser.add_argument("--enable-async-loading", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--post-request-settle-ms", type=float, default=250.0)
    args = parser.parse_args()

    os.environ["LMCACHE_ENABLE_FIDELITY_CACHE"] = "True"
    os.environ["LMCACHE_DEFAULT_FIDELITY"] = args.fidelity
    os.environ["LMCACHE_BASE_CODEC"] = args.base_codec
    os.environ["LMCACHE_USE_LAYERWISE"] = "True" if args.use_layerwise else "False"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "True" if args.enable_async_loading else "False"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 10**9)
    warm_tokens, query_tokens, data_meta = build_real_medium_tokens(
        tokenizer, Path(args.dataset_path), args.context_len, args.overlap_tokens
    )

    python_fallback = bool(getattr(lmc_ops, "PYTHON_FALLBACK", False))
    c_ops_available = not python_fallback
    config_row = {
        "event": "config",
        "workload": "longbench_narrativeqa_real_tokens",
        "model": args.model,
        "context_len": args.context_len,
        "reuse_bucket": "medium",
        "overlap_tokens": args.overlap_tokens,
        "fidelity": args.fidelity,
        "base_codec": args.base_codec,
        "use_layerwise": args.use_layerwise,
        "enable_async_loading": args.enable_async_loading,
        "max_tokens": args.max_tokens,
        "python_fallback": python_fallback,
        "c_ops_available": c_ops_available,
    }
    config_row.update(data_meta)
    print("PHASE1_CONFIG " + json.dumps(config_row, ensure_ascii=False), flush=True)

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        disable_log_stats=True,
        kv_transfer_config=KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both"),
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        warm_req_id = f"real_warm_{args.fidelity}_ctx{args.context_len}_medium"
        warm_result = await run_request(engine, warm_req_id, warm_tokens, args.max_tokens, args.fidelity)
        warm_row = {
            "event": "request_result",
            "request_kind": "warm",
            "workload": "longbench_narrativeqa_real_tokens",
            "model": args.model,
            "context_len": args.context_len,
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
        warm_row.update(data_meta)
        warm_row.update(warm_result)
        print("PHASE1_REQ " + json.dumps(warm_row, ensure_ascii=False), flush=True)
        if args.post_request_settle_ms > 0:
            await asyncio.sleep(args.post_request_settle_ms / 1000.0)

        req_id = f"real_{args.fidelity}_ctx{args.context_len}_medium"
        result = await run_request(engine, req_id, query_tokens, args.max_tokens, args.fidelity)
        row = {
            "event": "request_result",
            "request_kind": "measure",
            "workload": "longbench_narrativeqa_real_tokens",
            "model": args.model,
            "context_len": args.context_len,
            "reuse_bucket": "medium",
            "planned_overlap_tokens": args.overlap_tokens,
            "prompt_tokens": len(query_tokens),
            "requested_fidelity": args.fidelity,
            "base_codec": args.base_codec,
            "use_layerwise": args.use_layerwise,
            "enable_async_loading": args.enable_async_loading,
            "python_fallback": python_fallback,
            "c_ops_available": c_ops_available,
        }
        row.update(data_meta)
        row.update(result)
        print("PHASE1_REQ " + json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        engine.shutdown(timeout=0)


if __name__ == "__main__":
    asyncio.run(main())
