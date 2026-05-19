#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary_path = run_dir / "client" / "summary.json"
out = {"run_dir": str(run_dir)}
if summary_path.exists():
    out.update(json.loads(summary_path.read_text(encoding="utf-8")))
log_path = run_dir / "lmcache.log"
full_chunks = base_chunks = total_chunks = 0
ratios = []
if log_path.exists():
    pattern = re.compile(r"MP_MIXED_(?:STORE|RETRIEVE).*?chunks=(\d+).*?full_chunks=(\d+).*?base_chunks=(\d+).*?bytes_ratio=([0-9.]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            chunks, full, base, ratio = match.groups()
            total_chunks += int(chunks)
            full_chunks += int(full)
            base_chunks += int(base)
            ratios.append(float(ratio))
if ratios:
    out["lmcache_chunks"] = {
        "events": len(ratios),
        "chunks": total_chunks,
        "full_chunks": full_chunks,
        "base_chunks": base_chunks,
        "mean_event_bytes_ratio": round(sum(ratios) / len(ratios), 6),
        "weighted_bytes_ratio": round((full_chunks + 0.5078125 * base_chunks) / max(1, total_chunks), 6),
    }
(run_dir / "quality_run_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
