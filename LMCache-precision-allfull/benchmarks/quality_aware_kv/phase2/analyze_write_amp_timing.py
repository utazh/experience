#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REQID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")
LOG_TS_RE = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")
DUAL_WRITE_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] PHASE2_DUAL_WRITE(?: layer=(?P<layer>\d+))? "
    r"chunks=(?P<chunks>\d+) base_location=(?P<base_location>\S+) "
    r"full_location=(?P<full_location>\S+) base_encode_ms=(?P<base_encode_ms>[0-9.]+) "
    r"full_submit_ms=(?P<full_submit_ms>[0-9.]+) base_put_ms=(?P<base_put_ms>[0-9.]+)"
)
STORE_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] Stored (?P<stored>\d+) out of(?: total)? (?P<total>\d+) tokens\. "
    r"size: (?P<size_gb>[0-9.]+) GB, cost (?P<cost_ms>[0-9.]+) ms.*?"
    r"offload_time: (?P<offload_ms>[0-9.]+) ms, put_time: (?P<put_ms>[0-9.]+) ms"
)


def normalize_req_id(req_id: str) -> str:
    return REQID_SUFFIX_RE.sub("", req_id)


def parse_log_ts(line: str) -> float | None:
    match = LOG_TS_RE.search(line)
    if not match:
        return None
    dt = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
    return dt.timestamp() * 1000.0


def empty_timed_stats() -> dict[str, Any]:
    return {
        "events": 0,
        "first_log_ts_ms": None,
        "last_log_ts_ms": None,
    }


def update_ts(stats: dict[str, Any], ts_ms: float | None) -> None:
    if ts_ms is None:
        return
    if stats["first_log_ts_ms"] is None or ts_ms < stats["first_log_ts_ms"]:
        stats["first_log_ts_ms"] = ts_ms
    if stats["last_log_ts_ms"] is None or ts_ms > stats["last_log_ts_ms"]:
        stats["last_log_ts_ms"] = ts_ms


def finalize_timed_stats(stats: dict[str, Any]) -> dict[str, Any]:
    first_ts = stats.get("first_log_ts_ms")
    last_ts = stats.get("last_log_ts_ms")
    if first_ts is not None and last_ts is not None:
        stats["log_wall_span_ms"] = round(last_ts - first_ts, 3)
    else:
        stats["log_wall_span_ms"] = None
    return stats


def parse_phase2_log(log_path: Path) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    dual_write: dict[str, dict[str, Any]] = {}
    store: dict[str, dict[str, Any]] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("PHASE2_REQ "):
                requests.append(json.loads(line[len("PHASE2_REQ ") :]))
                continue

            ts_ms = parse_log_ts(line)
            match = DUAL_WRITE_RE.search(line)
            if match:
                req_id = normalize_req_id(match.group("req_id"))
                stats = dual_write.setdefault(
                    req_id,
                    {
                        **empty_timed_stats(),
                        "chunks": 0,
                        "base_encode_ms_sum": 0.0,
                        "full_submit_ms_sum": 0.0,
                        "base_put_ms_sum": 0.0,
                        "base_locations": set(),
                        "full_locations": set(),
                    },
                )
                stats["events"] += 1
                stats["chunks"] += int(match.group("chunks"))
                stats["base_encode_ms_sum"] += float(match.group("base_encode_ms"))
                stats["full_submit_ms_sum"] += float(match.group("full_submit_ms"))
                stats["base_put_ms_sum"] += float(match.group("base_put_ms"))
                stats["base_locations"].add(match.group("base_location"))
                stats["full_locations"].add(match.group("full_location"))
                update_ts(stats, ts_ms)
                continue

            match = STORE_RE.search(line)
            if match:
                req_id = normalize_req_id(match.group("req_id"))
                stats = store.setdefault(
                    req_id,
                    {
                        **empty_timed_stats(),
                        "stored_tokens": 0,
                        "store_total_tokens": 0,
                        "store_size_gb_sum": 0.0,
                        "store_cost_ms_sum": 0.0,
                        "store_offload_ms_sum": 0.0,
                        "store_put_ms_sum": 0.0,
                    },
                )
                stats["events"] += 1
                stats["stored_tokens"] += int(match.group("stored"))
                stats["store_total_tokens"] = max(
                    stats["store_total_tokens"], int(match.group("total"))
                )
                stats["store_size_gb_sum"] += float(match.group("size_gb"))
                stats["store_cost_ms_sum"] += float(match.group("cost_ms"))
                stats["store_offload_ms_sum"] += float(match.group("offload_ms"))
                stats["store_put_ms_sum"] += float(match.group("put_ms"))
                update_ts(stats, ts_ms)

    for stats in dual_write.values():
        stats["base_locations"] = ",".join(sorted(stats["base_locations"]))
        stats["full_locations"] = ",".join(sorted(stats["full_locations"]))
        stats["sync_sum_ms"] = (
            stats["base_encode_ms_sum"]
            + stats["full_submit_ms_sum"]
            + stats["base_put_ms_sum"]
        )
        finalize_timed_stats(stats)
    for stats in store.values():
        finalize_timed_stats(stats)

    return {
        "requests": requests,
        "dual_write": dual_write,
        "store": store,
    }


def select_request(
    parsed: dict[str, Any],
    *,
    scenario: str,
    request_kind: str,
    requested_fidelity: str | None = None,
) -> dict[str, Any]:
    for row in parsed["requests"]:
        if row.get("scenario") != scenario or row.get("request_kind") != request_kind:
            continue
        if requested_fidelity is not None and row.get("requested_fidelity") != requested_fidelity:
            continue
        return row
    raise ValueError(
        f"No request found for scenario={scenario!r}, request_kind={request_kind!r}, "
        f"requested_fidelity={requested_fidelity!r}"
    )


def request_stats(parsed: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    req_id = normalize_req_id(row.get("engine_req_id") or row["req_id"])
    return {
        "request": row,
        "store": parsed["store"].get(req_id, {}),
        "dual_write": parsed["dual_write"].get(req_id, {}),
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 3)


def infer_model_shape(model: str, chunk_tokens: int) -> dict[str, int]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    layers = int(getattr(cfg, "num_hidden_layers"))
    attention_heads = int(getattr(cfg, "num_attention_heads"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", attention_heads))
    hidden_size = int(getattr(cfg, "hidden_size"))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // attention_heads))
    return {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "chunk_tokens": chunk_tokens,
    }


def run_int8_microbench(
    *,
    model: str,
    chunk_tokens: int,
    chunks: int,
    repeats: int,
    dtype_name: str,
) -> dict[str, Any]:
    import torch

    shape_info = infer_model_shape(model, chunk_tokens)
    dtype = getattr(torch, dtype_name)
    shape = (
        shape_info["layers"],
        2,
        chunk_tokens,
        shape_info["kv_heads"],
        shape_info["head_dim"],
    )
    tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
    element_count = tensor.numel()
    input_bytes = element_count * tensor.element_size()

    encode_ms: list[float] = []
    encode_plus_copy_ms: list[float] = []

    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        working = tensor.to(torch.float32)
        scale = torch.clamp(working.abs().amax(dim=-1, keepdim=True) / 127.0, min=1e-8)
        quantized = torch.round(working / scale).clamp(-128, 127).to(torch.int8)
        encode_ms.append((time.perf_counter() - t0) * 1000.0)

        q_dst = torch.empty_like(quantized)
        s_dst = torch.empty_like(scale)
        t0 = time.perf_counter()
        working = tensor.to(torch.float32)
        scale = torch.clamp(working.abs().amax(dim=-1, keepdim=True) / 127.0, min=1e-8)
        quantized = torch.round(working / scale).clamp(-128, 127).to(torch.int8)
        q_dst.copy_(quantized)
        s_dst.copy_(scale)
        encode_plus_copy_ms.append((time.perf_counter() - t0) * 1000.0)

    encoded_bytes = quantized.numel() * quantized.element_size() + scale.numel() * scale.element_size()
    return {
        "model": model,
        "shape": shape_info,
        "dtype": dtype_name,
        "input_bytes_per_chunk": input_bytes,
        "encoded_bytes_per_chunk": encoded_bytes,
        "compression_ratio_per_chunk": round(encoded_bytes / input_bytes, 6),
        "chunks_for_estimate": chunks,
        "repeats": repeats,
        "encode_ms_per_chunk_mean": round(statistics.mean(encode_ms), 3),
        "encode_ms_per_chunk_p50": percentile(encode_ms, 0.5),
        "encode_plus_copy_ms_per_chunk_mean": round(statistics.mean(encode_plus_copy_ms), 3),
        "encode_plus_copy_ms_per_chunk_p50": percentile(encode_plus_copy_ms, 0.5),
        "estimated_encode_ms_total": round(statistics.mean(encode_ms) * chunks, 3),
        "estimated_encode_plus_copy_ms_total": round(
            statistics.mean(encode_plus_copy_ms) * chunks, 3
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    dual = parse_phase2_log(Path(args.dual_log))
    single = parse_phase2_log(Path(args.single_log))
    dual_row = select_request(
        dual,
        scenario=args.scenario,
        request_kind=args.request_kind,
        requested_fidelity=args.requested_fidelity,
    )
    single_row = select_request(
        single,
        scenario=args.scenario,
        request_kind=args.request_kind,
        requested_fidelity=args.requested_fidelity,
    )
    dual_stats = request_stats(dual, dual_row)
    single_stats = request_stats(single, single_row)
    dual_write = dual_stats["dual_write"]
    store_dual = dual_stats["store"]
    store_single = single_stats["store"]

    comparison = {
        "scenario": args.scenario,
        "request_kind": args.request_kind,
        "requested_fidelity": args.requested_fidelity,
        "dual_req_id": dual_row["req_id"],
        "single_req_id": single_row["req_id"],
        "dual_ttft_ms": dual_row.get("ttft_ms"),
        "single_ttft_ms": single_row.get("ttft_ms"),
        "ttft_overhead_ms": round(dual_row.get("ttft_ms") - single_row.get("ttft_ms"), 3),
        "dual_e2e_ms": dual_row.get("e2e_ms"),
        "single_e2e_ms": single_row.get("e2e_ms"),
        "e2e_overhead_ms": round(dual_row.get("e2e_ms") - single_row.get("e2e_ms"), 3),
        "dual_store_cost_ms_sum": round(store_dual.get("store_cost_ms_sum", 0.0), 3),
        "single_store_cost_ms_sum": round(store_single.get("store_cost_ms_sum", 0.0), 3),
        "store_cost_overhead_ms_sum": round(
            store_dual.get("store_cost_ms_sum", 0.0)
            - store_single.get("store_cost_ms_sum", 0.0),
            3,
        ),
        "dual_store_log_wall_span_ms": store_dual.get("log_wall_span_ms"),
        "single_store_log_wall_span_ms": store_single.get("log_wall_span_ms"),
        "dual_write_events": dual_write.get("events", 0),
        "dual_write_chunks": dual_write.get("chunks", 0),
        "dual_write_base_encode_ms_sum": round(
            dual_write.get("base_encode_ms_sum", 0.0), 3
        ),
        "dual_write_full_submit_ms_sum": round(
            dual_write.get("full_submit_ms_sum", 0.0), 3
        ),
        "dual_write_base_put_ms_sum": round(dual_write.get("base_put_ms_sum", 0.0), 3),
        "dual_write_sync_ms_sum": round(dual_write.get("sync_sum_ms", 0.0), 3),
        "dual_write_log_wall_span_ms": dual_write.get("log_wall_span_ms"),
        "dual_write_base_locations": dual_write.get("base_locations"),
        "dual_write_full_locations": dual_write.get("full_locations"),
    }

    microbench = None
    if args.run_microbench:
        chunks = args.microbench_chunks or max(int(dual_write.get("events", 0)), 1)
        microbench = run_int8_microbench(
            model=args.model,
            chunk_tokens=args.chunk_tokens,
            chunks=chunks,
            repeats=args.microbench_repeats,
            dtype_name=args.dtype,
        )

    interpretation = [
        "ttft_overhead_ms is a wall-clock delta between two complete requests/runs; it is the user-visible first-request cost.",
        "dual_write_base_encode_ms_sum is a sum of per-chunk CPU encode sections inside LMCache; it measures accumulated foreground work, not an independently measured critical-path delta versus the single-write run.",
        "single-write already pays prefill, D2H/offload, and base store costs; therefore dual_write_base_encode_ms_sum can be larger than ttft_overhead_ms without contradiction.",
        "Use TTFT overhead for user-visible latency, and report base_encode/full_submit/base_put sums as a decomposition of LMCache foreground work.",
    ]

    return {
        "artifact": "PHASE2_WRITE_AMP_TIMING_CLARIFICATION",
        "dual_log": args.dual_log,
        "single_log": args.single_log,
        "comparison": comparison,
        "microbench": microbench,
        "interpretation": interpretation,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    comp = report["comparison"]
    lines = [
        "# Phase 2 Write Amplification Timing Clarification",
        "",
        f"Dual log: `{report['dual_log']}`",
        "",
        f"Single log: `{report['single_log']}`",
        "",
        "## Log-Derived Comparison",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in comp.items():
        if key.endswith("_id") or key in {"scenario", "request_kind", "requested_fidelity"}:
            continue
        lines.append(f"| {key} | {value} |")
    if report.get("microbench"):
        lines.extend(["", "## Standalone CPU Int8 Microbench", "", "| metric | value |", "| --- | ---: |"])
        for key, value in report["microbench"].items():
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Interpretation"])
    for item in report["interpretation"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-log", required=True)
    parser.add_argument("--single-log", required=True)
    parser.add_argument("--scenario", default="no_promotion")
    parser.add_argument("--request-kind", default="prime")
    parser.add_argument("--requested-fidelity", default="base")
    parser.add_argument("--model", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--run-microbench", action="store_true")
    parser.add_argument("--microbench-repeats", type=int, default=1)
    parser.add_argument("--microbench-chunks", type=int, default=0)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    report = build_report(args)
    Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.md_out))


if __name__ == "__main__":
    main()
