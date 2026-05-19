#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_rows(path: str):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [row for row in rows if row.get("request_kind") == "measure"]


def key(row):
    return (int(row["context_len"]), str(row["reuse_bucket"]))


def get_retrieve_ms(row):
    return row.get("retrieve_ms")


def get_retrieve_gb(row):
    return row.get("retrieve_size_gb", row.get("retrieve_bytes_gb"))


def get_store_ms(row):
    return row.get("store_ms")


def get_store_gb(row):
    return row.get("store_size_gb", row.get("store_bytes_gb"))


def compare_output_match(row, ref_row):
    a = row.get("output_token_ids")
    b = ref_row.get("output_token_ids") if ref_row else None
    if not isinstance(a, list) or not isinstance(b, list) or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return round(sum(1 for i in range(n) if a[i] == b[i]) / n, 6)


def flatten_row(row, scenario, phase0_ref, full_ref):
    ttft_ms = row.get("ttft_ms")
    retrieve_ms = get_retrieve_ms(row)
    retrieve_gb = get_retrieve_gb(row)
    out = {
        "scenario": scenario,
        "context_len": row.get("context_len"),
        "reuse_bucket": row.get("reuse_bucket"),
        "requested_fidelity": row.get("requested_fidelity"),
        "use_layerwise": row.get("use_layerwise"),
        "enable_async_loading": row.get("enable_async_loading"),
        "python_fallback": row.get("python_fallback"),
        "c_ops_available": row.get("c_ops_available"),
        "ttft_ms": ttft_ms,
        "e2e_ms": row.get("e2e_ms"),
        "retrieve_ms": retrieve_ms,
        "retrieve_size_gb": retrieve_gb,
        "store_ms": get_store_ms(row),
        "store_size_gb": get_store_gb(row),
        "lmcache_hit_tokens": row.get("lmcache_hit_tokens"),
        "lmcache_hit_ratio": row.get("lmcache_hit_ratio"),
        "output_preview": row.get("output_preview"),
        "output_token_ids": row.get("output_token_ids"),
        "output_match_rate_vs_full_ref": compare_output_match(row, full_ref),
    }
    if phase0_ref and ttft_ms is not None and phase0_ref.get("ttft_ms") is not None:
        out["ttft_delta_vs_phase0_ms"] = round(ttft_ms - phase0_ref["ttft_ms"], 3)
        out["ttft_speedup_vs_phase0_x"] = round(phase0_ref["ttft_ms"] / ttft_ms, 6) if ttft_ms else None
    else:
        out["ttft_delta_vs_phase0_ms"] = None
        out["ttft_speedup_vs_phase0_x"] = None
    if full_ref and ttft_ms is not None and full_ref.get("ttft_ms") is not None:
        out["ttft_delta_vs_full_ref_ms"] = round(ttft_ms - full_ref["ttft_ms"], 3)
        out["ttft_speedup_vs_full_ref_x"] = round(full_ref["ttft_ms"] / ttft_ms, 6) if ttft_ms else None
    else:
        out["ttft_delta_vs_full_ref_ms"] = None
        out["ttft_speedup_vs_full_ref_x"] = None
    full_retrieve_ms = get_retrieve_ms(full_ref) if full_ref else None
    full_retrieve_gb = get_retrieve_gb(full_ref) if full_ref else None
    out["retrieve_ms_ratio_vs_full_ref"] = round(retrieve_ms / full_retrieve_ms, 6) if retrieve_ms and full_retrieve_ms else None
    out["retrieve_size_ratio_vs_full_ref"] = round(retrieve_gb / full_retrieve_gb, 6) if retrieve_gb and full_retrieve_gb else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-json", required=True)
    parser.add_argument("--full-json", required=True)
    parser.add_argument("--o1-json", required=True)
    parser.add_argument("--o2-json", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    phase0_rows = load_rows(args.phase0_json)
    full_rows = load_rows(args.full_json)
    o1_rows = load_rows(args.o1_json)
    o2_rows = load_rows(args.o2_json)

    phase0_by_key = {key(row): row for row in phase0_rows}
    full_by_key = {key(row): row for row in full_rows}

    merged = []
    for scenario, rows in [
        ("phase0_baseline", phase0_rows),
        ("full_reference", full_rows),
        ("o1_base_no_prefetch", o1_rows),
        ("o2_base_prefetch", o2_rows),
    ]:
        for row in rows:
            k = key(row)
            merged.append(
                flatten_row(
                    row=row,
                    scenario=scenario,
                    phase0_ref=phase0_by_key.get(k),
                    full_ref=full_by_key.get(k),
                )
            )

    merged.sort(key=lambda row: (row["context_len"], row["reuse_bucket"], row["scenario"]))
    Path(args.json_out).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({k for row in merged for k in row.keys()})
    with Path(args.csv_out).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


if __name__ == "__main__":
    main()
