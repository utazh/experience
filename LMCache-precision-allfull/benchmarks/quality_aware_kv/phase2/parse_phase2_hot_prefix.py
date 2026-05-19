#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REQID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")
RETRIEVE_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] Retrieved (?P<retrieved>\d+) out of (?P<required>\d+) required tokens "
    r"\(from (?P<total>\d+) total tokens\)\. size: (?P<size_gb>[0-9.]+) gb, cost (?P<cost_ms>[0-9.]+) ms"
)
STORE_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] Stored (?P<stored>\d+) out of(?: total)? (?P<total>\d+) tokens\. "
    r"size: (?P<size_gb>[0-9.]+) GB, cost (?P<cost_ms>[0-9.]+) ms"
)
DUAL_WRITE_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] PHASE2_DUAL_WRITE(?: layer=(?P<layer>\d+))? "
    r"chunks=(?P<chunks>\d+) base_location=(?P<base_location>\S+) "
    r"full_location=(?P<full_location>\S+) base_encode_ms=(?P<base_encode_ms>[0-9.]+) "
    r"full_submit_ms=(?P<full_submit_ms>[0-9.]+) base_put_ms=(?P<base_put_ms>[0-9.]+)"
)
BASE_CLEANUP_RE = re.compile(
    r"\[req_id=(?P<req_id>[^\]]+)\] PHASE2_BASE_CLEANUP chunks=(?P<chunks>\d+) "
    r"location=(?P<location>\S+) removed=(?P<removed>\d+)"
)
CORE_AUTO_RE = re.compile(
    r"PHASE2_CORE_AUTO_DECISION prefix_id=(?P<prefix_id>\S+) "
    r"state=(?P<state>\S+) (?:sequence_index=(?P<sequence_index>\S+) )?"
    r"selected_fidelity=(?P<selected_fidelity>\S+) reason=(?P<reason>\S+)"
)
REQ_STATS_RE = re.compile(
    r"Reqid: (?P<req_id>[^,]+), Total tokens (?P<total>\d+), Inference Engine computed tokens: "
    r"(?P<computed>\d+), LMCache hit tokens: (?P<hit>\d+), need to load: (?P<need>\d+)"
)


def normalize_req_id(req_id: str) -> str:
    return REQID_SUFFIX_RE.sub("", req_id)


def load_log(log_path: Path):
    configs = []
    requests = []
    states = []
    summaries = []
    auto_decisions = []
    core_auto_decisions = []
    retrieve_stats = {}
    store_stats = {}
    dual_write_stats = {}
    cleanup_stats = {}
    req_stats = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("PHASE2_CONFIG "):
                configs.append(json.loads(line[len("PHASE2_CONFIG "):]))
                continue
            if line.startswith("PHASE2_MIXED_CONFIG "):
                configs.append(json.loads(line[len("PHASE2_MIXED_CONFIG "):]))
                continue
            if line.startswith("PHASE2_REQ "):
                requests.append(json.loads(line[len("PHASE2_REQ "):]))
                continue
            if line.startswith("PHASE2_STATE "):
                states.append(json.loads(line[len("PHASE2_STATE "):]))
                continue
            if line.startswith("PHASE2_AUTO_DECISION "):
                auto_decisions.append(json.loads(line[len("PHASE2_AUTO_DECISION "):]))
                continue
            if line.startswith("PHASE2_SUMMARY "):
                summaries.append(json.loads(line[len("PHASE2_SUMMARY "):]))
                continue
            if line.startswith("PHASE2_MIXED_SUMMARY "):
                summaries.append(json.loads(line[len("PHASE2_MIXED_SUMMARY "):]))
                continue
            m = CORE_AUTO_RE.search(line)
            if m:
                sequence_index_raw = m.group("sequence_index")
                sequence_index = None
                if sequence_index_raw and sequence_index_raw != "None":
                    sequence_index = int(sequence_index_raw)
                core_auto_decisions.append(
                    {
                        "event": "core_auto_policy_decision",
                        "prefix_id": m.group("prefix_id"),
                        "state": m.group("state"),
                        "sequence_index": sequence_index,
                        "selected_fidelity": m.group("selected_fidelity"),
                        "reason": m.group("reason"),
                    }
                )
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
                stats = retrieve_stats.setdefault(req_id, {
                    "retrieved_tokens": 0,
                    "required_tokens": 0,
                    "retrieve_total_tokens": 0,
                    "retrieve_size_gb": 0.0,
                    "retrieve_ms": 0.0,
                    "retrieve_events": 0,
                })
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
                stats = store_stats.setdefault(req_id, {
                    "stored_tokens": 0,
                    "store_total_tokens": 0,
                    "store_size_gb": 0.0,
                    "store_ms": 0.0,
                    "store_events": 0,
                })
                stats["stored_tokens"] += int(m.group("stored"))
                stats["store_total_tokens"] = max(stats["store_total_tokens"], int(m.group("total")))
                stats["store_size_gb"] += float(m.group("size_gb"))
                stats["store_ms"] += float(m.group("cost_ms"))
                stats["store_events"] += 1
                continue
            m = DUAL_WRITE_RE.search(line)
            if m:
                req_id = normalize_req_id(m.group("req_id"))
                stats = dual_write_stats.setdefault(req_id, {
                    "dual_write_chunks": 0,
                    "dual_write_events": 0,
                    "dual_write_base_encode_ms": 0.0,
                    "dual_write_full_submit_ms": 0.0,
                    "dual_write_base_put_ms": 0.0,
                    "dual_write_base_locations": set(),
                    "dual_write_full_locations": set(),
                })
                stats["dual_write_chunks"] += int(m.group("chunks"))
                stats["dual_write_events"] += 1
                stats["dual_write_base_encode_ms"] += float(m.group("base_encode_ms"))
                stats["dual_write_full_submit_ms"] += float(m.group("full_submit_ms"))
                stats["dual_write_base_put_ms"] += float(m.group("base_put_ms"))
                stats["dual_write_base_locations"].add(m.group("base_location"))
                stats["dual_write_full_locations"].add(m.group("full_location"))
                continue
            m = BASE_CLEANUP_RE.search(line)
            if m:
                req_id = normalize_req_id(m.group("req_id"))
                stats = cleanup_stats.setdefault(req_id, {
                    "base_cleanup_chunks": 0,
                    "base_cleanup_removed": 0,
                    "base_cleanup_events": 0,
                    "base_cleanup_locations": set(),
                })
                stats["base_cleanup_chunks"] += int(m.group("chunks"))
                stats["base_cleanup_removed"] += int(m.group("removed"))
                stats["base_cleanup_events"] += 1
                stats["base_cleanup_locations"].add(m.group("location"))

    merged = []
    for stats in dual_write_stats.values():
        stats["dual_write_base_locations"] = ",".join(sorted(stats["dual_write_base_locations"]))
        stats["dual_write_full_locations"] = ",".join(sorted(stats["dual_write_full_locations"]))
    for stats in cleanup_stats.values():
        stats["base_cleanup_locations"] = ",".join(sorted(stats["base_cleanup_locations"]))
    for row in requests:
        engine_req_id = normalize_req_id(row.get("engine_req_id") or row["req_id"])
        row_req_id = normalize_req_id(row["req_id"])
        merged_row = dict(row)
        merged_row.update(req_stats.get(engine_req_id, req_stats.get(row_req_id, {})))
        merged_row.update(retrieve_stats.get(engine_req_id, retrieve_stats.get(row_req_id, {})))
        merged_row.update(store_stats.get(engine_req_id, store_stats.get(row_req_id, {})))
        merged_row.update(dual_write_stats.get(engine_req_id, dual_write_stats.get(row_req_id, {})))
        merged_row.update(cleanup_stats.get(engine_req_id, cleanup_stats.get(row_req_id, {})))
        if merged_row.get("fidelity_policy_path") == "core_auto_state_store_v0":
            matches = [
                decision
                for decision in core_auto_decisions
                if decision.get("prefix_id") == merged_row.get("prefix_id")
                and decision.get("state") == merged_row.get("state_before")
                and decision.get("sequence_index") == merged_row.get("sequence_index")
                and decision.get("selected_fidelity") == merged_row.get("requested_fidelity")
            ]
            if matches:
                merged_row["core_auto_selected_fidelity"] = matches[0]["selected_fidelity"]
                merged_row["core_auto_reason"] = matches[0]["reason"]
            merged_row["core_auto_decision_count"] = len(matches)
            merged_row["core_auto_match"] = bool(matches)
        if merged_row.get("lmcache_total_tokens"):
            merged_row["lmcache_hit_ratio"] = round(
                merged_row.get("lmcache_hit_tokens", 0) / merged_row["lmcache_total_tokens"], 6
            )
        merged.append(merged_row)
    return configs, merged, states, summaries, auto_decisions, core_auto_decisions


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--requests-json", required=True)
    parser.add_argument("--requests-csv", required=True)
    parser.add_argument("--states-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    configs, requests, states, summaries, auto_decisions, core_auto_decisions = load_log(Path(args.log))
    Path(args.requests_json).write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.requests_csv), requests)
    Path(args.states_json).write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_payload = {
        "configs": configs,
        "summaries": summaries,
        "auto_decisions": auto_decisions,
        "core_auto_decisions": core_auto_decisions,
    }
    Path(args.summary_json).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.summary_csv), summaries)


if __name__ == "__main__":
    main()
