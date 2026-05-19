#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import statistics
import string
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


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def score_prediction(prediction: str, answers: list[str]) -> dict[str, float]:
    return {
        "f1": metric_max_over_ground_truths(f1_score, prediction, answers),
        "em": metric_max_over_ground_truths(exact_match_score, prediction, answers),
    }


def fit_prompt_tokens(tokenizer, prefix_text: str, suffix_text: str, context_len: int) -> list[int]:
    suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)
    if len(suffix_tokens) >= context_len:
        return suffix_tokens[-context_len:]
    prefix_budget = context_len - len(suffix_tokens)
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    return (prefix_tokens[:prefix_budget] + suffix_tokens)[:context_len]


def build_narrativeqa_prompt(row: dict[str, Any]) -> tuple[str, str]:
    context = normalize_text(row.get("context"))
    question = normalize_text(row.get("input") or row.get("question"))
    prefix_text = (
        "You are given a long document from NarrativeQA. "
        "Answer the question using only the document. Keep the answer short.\n\n"
        f"Document:\n{context}\n\n"
    )
    suffix_text = f"Question:\n{question}\n\nAnswer:"
    return prefix_text, suffix_text


def load_narrativeqa_samples(tokenizer, path: Path, context_len: int, count: int, start_index: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row_index, row in enumerate(load_jsonl(path)):
        if row_index < start_index:
            continue
        answers = [normalize_text(a) for a in (row.get("answers") or []) if normalize_text(a).strip()]
        if not answers:
            continue
        prefix_text, suffix_text = build_narrativeqa_prompt(row)
        tokens = fit_prompt_tokens(tokenizer, prefix_text, suffix_text, context_len)
        if len(tokens) < context_len:
            continue
        samples.append({
            "sample_index": len(samples),
            "row_index": row_index,
            "record_id": row.get("_id"),
            "dataset": row.get("dataset", "narrativeqa"),
            "question": normalize_text(row.get("input") or row.get("question")),
            "answers": answers,
            "source_length": row.get("length"),
            "prompt_tokens": tokens,
            "context_len": len(tokens),
            "construction": "LongBench NarrativeQA context/question; context is head-truncated when needed while preserving the question suffix.",
        })
        if len(samples) >= count:
            break
    if len(samples) < count:
        raise ValueError(f"Need {count} NarrativeQA samples with >= {context_len} tokens, got {len(samples)} from {path}")
    return samples


async def run_request(
    engine: AsyncLLM,
    req_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    fidelity: str,
    skip_save: bool,
    stop_at_newline: bool,
) -> dict[str, Any]:
    kv_params: dict[str, Any] = {"lmcache.fidelity": fidelity}
    if skip_save:
        kv_params["lmcache.skip_save"] = True
    sampling_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "output_kind": RequestOutputKind.DELTA,
        "extra_args": {"kv_transfer_params": kv_params},
    }
    if stop_at_newline:
        sampling_kwargs["stop"] = ["\n"]
    params = SamplingParams(**sampling_kwargs)
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
            token_ids = getattr(sample, "token_ids", None)
            if token_ids:
                output_token_ids.extend(list(token_ids))
    end = time.perf_counter()
    return {
        "req_id": req_id,
        "engine_req_id": engine_req_id,
        "start_wall_ts": start_wall_ts,
        "first_token_wall_ts": first_wall_ts,
        "end_wall_ts": now_str(),
        "ttft_ms": None if first is None else round((first - start) * 1000.0, 3),
        "e2e_ms": round((end - start) * 1000.0, 3),
        "prediction": final_text.strip(),
        "output_preview": final_text[:240],
        "output_token_count": len(output_token_ids),
        "output_token_ids": output_token_ids,
    }


def emit_result(row: dict[str, Any]) -> None:
    print("NARRATIVEQA_QA_ROW " + json.dumps(row, ensure_ascii=False), flush=True)


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 6) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    modes = sorted({row["mode"] for row in rows if row.get("scored")})
    for mode in modes:
        subset = [row for row in rows if row.get("mode") == mode and row.get("scored")]
        summaries.append({
            "mode": mode,
            "sample_count": len(subset),
            "mean_f1": mean([row["f1"] for row in subset]),
            "mean_em": mean([row["em"] for row in subset]),
            "median_f1": median([row["f1"] for row in subset]),
            "median_em": median([row["em"] for row in subset]),
            "mean_ttft_ms": mean([row["ttft_ms"] for row in subset if row.get("ttft_ms") is not None]),
            "mean_e2e_ms": mean([row["e2e_ms"] for row in subset if row.get("e2e_ms") is not None]),
        })
    by_mode = {row["mode"]: row for row in summaries}
    if "base_cached" in by_mode and "full_cached" in by_mode:
        by_mode["base_cached"]["delta_mean_f1_vs_full_cached"] = round(
            by_mode["base_cached"]["mean_f1"] - by_mode["full_cached"]["mean_f1"], 6
        )
        by_mode["base_cached"]["delta_mean_em_vs_full_cached"] = round(
            by_mode["base_cached"]["mean_em"] - by_mode["full_cached"]["mean_em"], 6
        )
    if "base_cached" in by_mode and "reference_uncached" in by_mode:
        by_mode["base_cached"]["delta_mean_f1_vs_reference_uncached"] = round(
            by_mode["base_cached"]["mean_f1"] - by_mode["reference_uncached"]["mean_f1"], 6
        )
        by_mode["base_cached"]["delta_mean_em_vs_reference_uncached"] = round(
            by_mode["base_cached"]["mean_em"] - by_mode["reference_uncached"]["mean_em"], 6
        )
    return summaries


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-path", default="/data1/datasets/LongBench/data/narrativeqa.jsonl")
    parser.add_argument("--context-len", type=int, default=16384)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--base-codec", default="int8", choices=["fake", "int8"])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stop-at-newline", action="store_true")
    parser.add_argument("--post-request-settle-ms", type=float, default=100.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-local-cpu-size", type=float, default=20.0)
    parser.add_argument("--local-disk", required=True)
    parser.add_argument("--max-local-disk-size", type=float, default=80.0)
    args = parser.parse_args()

    os.environ["LMCACHE_ENABLE_FIDELITY_CACHE"] = "True"
    os.environ["LMCACHE_DEFAULT_FIDELITY"] = "full"
    os.environ["LMCACHE_ENABLE_FIDELITY_INTERNAL_STATE"] = "False"
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
    samples = load_narrativeqa_samples(tokenizer, Path(args.dataset_path), args.context_len, args.num_samples, args.start_index)
    max_model_len = args.max_model_len or (args.context_len + args.max_tokens + 512)
    python_fallback = bool(getattr(lmc_ops, "PYTHON_FALLBACK", False))
    config = {
        "event": "config",
        "workload": "narrativeqa_f1_em_cached_fidelity_quality",
        "model": args.model,
        "dataset": "LongBench/narrativeqa",
        "dataset_path": args.dataset_path,
        "context_len": args.context_len,
        "num_samples": args.num_samples,
        "start_index": args.start_index,
        "max_tokens": args.max_tokens,
        "stop_at_newline": args.stop_at_newline,
        "base_codec": args.base_codec,
        "python_fallback": python_fallback,
        "c_ops_available": not python_fallback,
        "store_base_target": os.environ.get("LMCACHE_STORE_BASE_TARGET"),
        "store_full_target": os.environ.get("LMCACHE_STORE_FULL_TARGET"),
        "cleanup_base_on_full_store": os.environ.get("LMCACHE_CLEANUP_BASE_ON_FULL_STORE", "False"),
        "quality_boundary": "Scores are computed on cached follow-up generations. reference_uncached uses full fidelity with skip_save; base_cached primes then retrieves INT8 base KV; full_cached primes then retrieves full KV. Base cleanup is disabled for this A/B quality run so base and full variants can coexist for the same prompts.",
    }
    print("NARRATIVEQA_QA_CONFIG " + json.dumps(config, ensure_ascii=False), flush=True)

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
        for sample in samples:
            prompt_tokens = sample["prompt_tokens"]
            base_meta = {k: v for k, v in sample.items() if k != "prompt_tokens"}
            request_plan = [
                ("reference_uncached", "reference", "full", True, True),
                ("base_prime", "prime", "base", False, False),
                ("base_cached", "measure", "base", False, True),
                ("full_prime", "prime", "full", False, False),
                ("full_cached", "measure", "full", False, True),
            ]
            for mode, request_kind, fidelity, skip_save, scored in request_plan:
                req_id = f"narrativeqa_{sample['sample_index']}_{mode}_{fidelity}"
                result = await run_request(
                    engine,
                    req_id,
                    prompt_tokens,
                    args.max_tokens,
                    fidelity,
                    skip_save,
                    args.stop_at_newline,
                )
                row = {
                    "event": "qa_result",
                    "mode": mode,
                    "request_kind": request_kind,
                    "requested_fidelity": fidelity,
                    "skip_save": skip_save,
                    "scored": scored,
                }
                row.update(base_meta)
                row.update(result)
                if scored:
                    row.update(score_prediction(row["prediction"], sample["answers"]))
                emit_result(row)
                rows.append(row)
                if args.post_request_settle_ms > 0:
                    await asyncio.sleep(args.post_request_settle_ms / 1000.0)
    finally:
        engine.shutdown(timeout=0)

    summary = {
        "event": "summary",
        "workload": "narrativeqa_f1_em_cached_fidelity_quality",
        "model": args.model,
        "dataset_path": args.dataset_path,
        "context_len": args.context_len,
        "num_samples": args.num_samples,
        "summaries": summarize(rows),
    }
    print("NARRATIVEQA_QA_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
