
#!/usr/bin/env python3
import argparse
import csv
import json
import re

REQID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")
from pathlib import Path

RETRIEVE_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] Retrieved (?P<retrieved>\d+) out of (?P<required>\d+) required tokens "
    r"\(from (?P<total>\d+) total tokens\)\. size: (?P<size_gb>[0-9.]+) gb, cost (?P<cost_ms>[0-9.]+) ms"
)
STORE_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] Stored (?P<stored>\d+) out of(?: total)? (?P<total>\d+) tokens\. "
    r"size: (?P<size_gb>[0-9.]+) GB, cost (?P<cost_ms>[0-9.]+) ms"
)
REQ_STATS_RE = re.compile(
    r"Reqid: (?P<req_id>[^,]+), Total tokens (?P<total>\d+), Inference Engine computed tokens: "
    r"(?P<computed>\d+), LMCache hit tokens: (?P<hit>\d+), need to load: (?P<need>\d+)"
)


def normalize_req_id(req_id: str) -> str:
    return REQID_SUFFIX_RE.sub("", req_id)


def load_rows(log_path: Path):
    rows = []
    retrieve_stats = {}
    store_stats = {}
    req_stats = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("PHASE1_REQ "):
                rows.append(json.loads(line[len("PHASE1_REQ "):]))
                continue
            m = REQ_STATS_RE.search(line)
            if m:
                req_id = normalize_req_id(m.group("req_id"))
                req_stats[req_id] = {
                    "lmcache_total_tokens": int(m.group("total")),
                    "lmcache_computed_tokens": int(m.group("computed")),
                    "lmcache_hit_tokens": int(m.group("hit")),
                    "lmcache_need_to_load_tokens": int(m.group("need")),
                }
                continue
            m = RETRIEVE_RE.search(line)
            if m:
                req_id = normalize_req_id(m.group("req_id"))
                stats = retrieve_stats.setdefault(
                    req_id,
                    {
                        "retrieved_tokens": 0,
                        "required_tokens": 0,
                        "retrieve_total_tokens": 0,
                        "retrieve_size_gb": 0.0,
                        "retrieve_ms": 0.0,
                        "retrieve_events": 0,
                    },
                )
                stats["retrieved_tokens"] += int(m.group("retrieved"))
                stats["required_tokens"] = max(stats["required_tokens"], int(m.group("required")))
                stats["retrieve_total_tokens"] = max(stats["retrieve_total_tokens"], int(m.group("total")))
                stats["retrieve_size_gb"] += float(m.group("size_gb"))
                stats["retrieve_ms"] += float(m.group("cost_ms"))
                stats["retrieve_events"] += 1
                continue
            m = STORE_RE.search(line)
            if m:
                req_id = normalize_req_id(m.group("req_id"))
                stats = store_stats.setdefault(
                    req_id,
                    {
                        "stored_tokens": 0,
                        "store_total_tokens": 0,
                        "store_size_gb": 0.0,
                        "store_ms": 0.0,
                        "store_events": 0,
                    },
                )
                stats["stored_tokens"] += int(m.group("stored"))
                stats["store_total_tokens"] = max(stats["store_total_tokens"], int(m.group("total")))
                stats["store_size_gb"] += float(m.group("size_gb"))
                stats["store_ms"] += float(m.group("cost_ms"))
                stats["store_events"] += 1
    merged = []
    for row in rows:
        engine_req_id = normalize_req_id(row.get("engine_req_id") or row["req_id"])
        row_req_id = normalize_req_id(row["req_id"])
        merged_row = dict(row)
        merged_row.update(req_stats.get(engine_req_id, req_stats.get(row_req_id, {})))
        merged_row.update(retrieve_stats.get(engine_req_id, retrieve_stats.get(row_req_id, {})))
        merged_row.update(store_stats.get(engine_req_id, store_stats.get(row_req_id, {})))
        if merged_row.get("lmcache_total_tokens"):
            merged_row["lmcache_hit_ratio"] = round(
                merged_row.get("lmcache_hit_tokens", 0) / merged_row["lmcache_total_tokens"], 6
            )
        merged.append(merged_row)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.log))
    Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with Path(args.csv_out).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        Path(args.csv_out).write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
