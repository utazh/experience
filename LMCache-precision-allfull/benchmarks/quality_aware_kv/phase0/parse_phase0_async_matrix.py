#!/usr/bin/env python3
import csv
import json
import os
import re
import sys
from datetime import datetime

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")
RE_REQ = re.compile(r"^PHASE0_REQ\s+(\{.*\})$")
RE_HIT = re.compile(
    r"Reqid:\s+(?P<req_id>[^,]+),\s+Total tokens\s+(?P<total>\d+),\s+Inference Engine computed tokens:\s+(?P<computed>\d+),\s+LMCache hit tokens:\s+(?P<hit>\d+),\s+need to load:\s+(?P<need>\d+)"
)
RE_RETRIEVE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\]\s+Retrieved\s+(?P<retrieved>\d+)\s+out of\s+(?P<required>\d+)\s+required tokens\s+\(from\s+(?P<total>\d+)\s+total tokens\)\.\s+size:\s+(?P<size_gb>[0-9.]+)\s+gb,\s+cost\s+(?P<cost_ms>[0-9.]+)\s+ms"
)
RE_STORE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\]\s+Stored\s+(?P<stored>\d+)\s+out of total\s+(?P<total>\d+)\s+tokens\.\s+size:\s+(?P<size_gb>[0-9.]+)\s+GB,\s+cost\s+(?P<cost_ms>[0-9.]+)\s+ms"
)
REQ_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S,%f"


def clean(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def parse_log_ts(line: str):
    m = TS_RE.search(line)
    if not m:
        return None
    return datetime.strptime(m.group("ts"), LOG_TS_FMT)


def parse_req_ts(value):
    if not value:
        return None
    return datetime.strptime(value, REQ_TS_FMT)


def diff_ms(later, earlier):
    if later is None or earlier is None:
        return None
    return round((later - earlier).total_seconds() * 1000.0, 3)


def to_int(m, key):
    return int(m.group(key))


def to_float(m, key):
    return float(m.group(key))


def add_sum(target: dict, key: str, value):
    target[key] = round(target.get(key, 0.0) + value, 6) if isinstance(value, float) else target.get(key, 0) + value


def resolve_logged_req_id(logged_req_id: str, known_req_ids: list[str]) -> str | None:
    if logged_req_id in known_req_ids:
        return logged_req_id
    for req_id in sorted(known_req_ids, key=len, reverse=True):
        if logged_req_id.startswith(req_id + "-"):
            return req_id
    return None


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: parse_phase0_async_matrix.py <log_path> <out_dir>")
    log_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    req_rows = {}
    raw_hit_stats = {}
    raw_retrieve_stats = {}
    raw_store_stats = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = clean(raw_line)
            if not line:
                continue
            line_ts = parse_log_ts(line)
            m = RE_REQ.match(line)
            if m:
                row = json.loads(m.group(1))
                req_rows[row["req_id"]] = row
                continue
            m = RE_HIT.search(line)
            if m:
                raw_hit_stats[m.group("req_id")] = {
                    "lmcache_total_tokens": to_int(m, "total"),
                    "lmcache_computed_tokens": to_int(m, "computed"),
                    "lmcache_hit_tokens": to_int(m, "hit"),
                    "lmcache_need_to_load_tokens": to_int(m, "need"),
                    "hit_line_wall_ts": None if line_ts is None else line_ts.strftime(REQ_TS_FMT),
                }
                continue
            m = RE_RETRIEVE.search(line)
            if m:
                stat = raw_retrieve_stats.setdefault(m.group("req_id"), {
                    "retrieve_tokens": 0,
                    "retrieve_required_tokens": 0,
                    "retrieve_total_tokens": 0,
                    "retrieve_bytes_gb": 0.0,
                    "retrieve_ms": 0.0,
                    "retrieve_events": 0,
                    "retrieve_first_wall_ts": None,
                    "retrieve_last_wall_ts": None,
                })
                add_sum(stat, "retrieve_tokens", to_int(m, "retrieved"))
                add_sum(stat, "retrieve_required_tokens", to_int(m, "required"))
                add_sum(stat, "retrieve_total_tokens", to_int(m, "total"))
                add_sum(stat, "retrieve_bytes_gb", to_float(m, "size_gb"))
                add_sum(stat, "retrieve_ms", to_float(m, "cost_ms"))
                add_sum(stat, "retrieve_events", 1)
                ts_s = None if line_ts is None else line_ts.strftime(REQ_TS_FMT)
                if stat["retrieve_first_wall_ts"] is None:
                    stat["retrieve_first_wall_ts"] = ts_s
                stat["retrieve_last_wall_ts"] = ts_s
                continue
            m = RE_STORE.search(line)
            if m:
                stat = raw_store_stats.setdefault(m.group("req_id"), {
                    "store_tokens": 0,
                    "store_total_tokens": 0,
                    "store_bytes_gb": 0.0,
                    "store_ms": 0.0,
                    "store_events": 0,
                })
                add_sum(stat, "store_tokens", to_int(m, "stored"))
                add_sum(stat, "store_total_tokens", to_int(m, "total"))
                add_sum(stat, "store_bytes_gb", to_float(m, "size_gb"))
                add_sum(stat, "store_ms", to_float(m, "cost_ms"))
                add_sum(stat, "store_events", 1)
                continue

    known_req_ids = list(req_rows.keys())
    hit_stats = {}
    retrieve_stats = {}
    store_stats = {}
    unresolved_logged_req_ids = []

    for logged_req_id, stat in raw_hit_stats.items():
        resolved = resolve_logged_req_id(logged_req_id, known_req_ids)
        if resolved is None:
            unresolved_logged_req_ids.append(logged_req_id)
            continue
        hit_stats[resolved] = dict(stat)
        hit_stats[resolved]["logged_req_id"] = logged_req_id

    for logged_req_id, stat in raw_retrieve_stats.items():
        resolved = resolve_logged_req_id(logged_req_id, known_req_ids)
        if resolved is None:
            unresolved_logged_req_ids.append(logged_req_id)
            continue
        retrieve_stats[resolved] = dict(stat)
        retrieve_stats[resolved]["logged_req_id"] = logged_req_id

    for logged_req_id, stat in raw_store_stats.items():
        resolved = resolve_logged_req_id(logged_req_id, known_req_ids)
        if resolved is None:
            unresolved_logged_req_ids.append(logged_req_id)
            continue
        store_stats[resolved] = dict(stat)
        store_stats[resolved]["logged_req_id"] = logged_req_id

    rows = []
    for req_id, row in req_rows.items():
        merged = dict(row)
        if req_id in hit_stats:
            merged.update(hit_stats[req_id])
        if req_id in retrieve_stats:
            merged.update(retrieve_stats[req_id])
        if req_id in store_stats:
            merged.update(store_stats[req_id])
        if "engine_req_id" not in merged and "logged_req_id" in merged:
            merged["engine_req_id"] = merged["logged_req_id"]

        hit_tokens = merged.get("lmcache_hit_tokens")
        ctx_len = merged.get("context_len") or merged.get("prompt_tokens")
        retrieve_ms = merged.get("retrieve_ms")
        ttft_ms = merged.get("ttft_ms")
        hit_dt = parse_req_ts(merged.get("hit_line_wall_ts"))
        retrieve_last_dt = parse_req_ts(merged.get("retrieve_last_wall_ts"))
        first_token_dt = parse_req_ts(merged.get("first_token_wall_ts"))

        merged["lmcache_hit_ratio"] = None if hit_tokens is None or not ctx_len else round(hit_tokens / ctx_len, 6)
        merged["kv_related_stall_ms"] = retrieve_ms if retrieve_ms is not None else 0.0
        merged["kv_stall_ratio"] = None if ttft_ms in (None, 0) else round((merged["kv_related_stall_ms"] or 0.0) / ttft_ms, 6)
        merged["lmcache_path_wall_ms"] = diff_ms(retrieve_last_dt, hit_dt)
        merged["post_retrieve_to_first_token_ms"] = diff_ms(first_token_dt, retrieve_last_dt)
        merged["non_kv_ttft_ms"] = None if ttft_ms is None else round(ttft_ms - (retrieve_ms or 0.0), 3)
        if ttft_ms is None or merged["lmcache_path_wall_ms"] is None:
            merged["residual_after_lmcache_path_ms"] = None
        else:
            merged["residual_after_lmcache_path_ms"] = round(ttft_ms - merged["lmcache_path_wall_ms"], 3)
        merged.setdefault("gpu_prefill_ms", None)
        merged.setdefault("lookup_ms", None)
        merged.setdefault("process_tokens_ms", None)
        merged.setdefault("contains_ms", None)
        merged.setdefault("backend_get_wait_ms", None)
        merged.setdefault("cpu_to_gpu_ms", None)
        merged.setdefault("backend_read_bytes_gb", merged.get("retrieve_bytes_gb"))
        rows.append(merged)

    rows.sort(key=lambda x: (x["context_len"], x["request_kind"], x["reuse_bucket"], x["req_id"]))

    summary_path = os.path.join(out_dir, "phase0_matrix_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "phase0_matrix_results.csv")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    measure_rows = [r for r in rows if r.get("request_kind") == "measure"]
    grouped = []
    for row in measure_rows:
        grouped.append({
            "context_len": row["context_len"],
            "reuse_bucket": row["reuse_bucket"],
            "ttft_ms": row.get("ttft_ms"),
            "e2e_ms": row.get("e2e_ms"),
            "lmcache_hit_tokens": row.get("lmcache_hit_tokens"),
            "lmcache_hit_ratio": row.get("lmcache_hit_ratio"),
            "retrieve_ms": row.get("retrieve_ms"),
            "lmcache_path_wall_ms": row.get("lmcache_path_wall_ms"),
            "post_retrieve_to_first_token_ms": row.get("post_retrieve_to_first_token_ms"),
            "non_kv_ttft_ms": row.get("non_kv_ttft_ms"),
            "store_ms": row.get("store_ms"),
            "kv_stall_ratio": row.get("kv_stall_ratio"),
        })
    grouped_path = os.path.join(out_dir, "phase0_matrix_measure_summary.json")
    with open(grouped_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "summary_path": summary_path,
        "csv_path": csv_path,
        "grouped_path": grouped_path,
        "row_count": len(rows),
        "measure_count": len(measure_rows),
        "unresolved_logged_req_ids": sorted(set(unresolved_logged_req_ids)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
