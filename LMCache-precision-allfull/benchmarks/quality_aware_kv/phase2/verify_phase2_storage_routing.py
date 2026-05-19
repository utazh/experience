#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DUAL_WRITE_RE = re.compile(
    r"PHASE2_DUAL_WRITE(?: layer=(?P<layer>\d+))? chunks=(?P<chunks>\d+) "
    r"base_location=(?P<base_location>\S+) full_location=(?P<full_location>\S+)"
)
DISK_DIR_RE = re.compile(r"^DISK_DIR=(?P<value>.+)$")


def _strip_shell_value(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _resolve_disk_dir(run_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    run_sh = run_dir / "run.sh"
    if not run_sh.exists():
        return None
    for line in run_sh.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DISK_DIR_RE.match(line.strip())
        if match:
            return Path(_strip_shell_value(match.group("value")))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log", default="run.log")
    parser.add_argument("--disk-dir")
    parser.add_argument("--expect-base-location", default="LocalCPUBackend")
    parser.add_argument("--expect-full-location", default="LocalDiskBackend")
    parser.add_argument("--require-full-disk-files", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = run_dir / log_path
    disk_dir = _resolve_disk_dir(run_dir, args.disk_dir)

    events = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = DUAL_WRITE_RE.search(line)
            if not match:
                continue
            events.append(match.groupdict())

    base_locations = sorted({event["base_location"] for event in events})
    full_locations = sorted({event["full_location"] for event in events})
    disk_files = []
    full_disk_files = []
    base_disk_files = []
    if disk_dir is not None and disk_dir.exists():
        disk_files = [p for p in disk_dir.rglob("*") if p.is_file()]
        full_disk_files = [p for p in disk_files if "fidelity%full" in p.name]
        base_disk_files = [p for p in disk_files if "fidelity%base" in p.name]

    errors = []
    if not events:
        errors.append("no PHASE2_DUAL_WRITE events found")
    if base_locations != [args.expect_base_location]:
        errors.append(f"base locations {base_locations} != {[args.expect_base_location]}")
    if full_locations != [args.expect_full_location]:
        errors.append(f"full locations {full_locations} != {[args.expect_full_location]}")
    if args.require_full_disk_files and not full_disk_files:
        errors.append("no full fidelity files found in disk dir")
    if base_disk_files:
        errors.append(f"found base fidelity files in disk dir: {len(base_disk_files)}")

    payload = {
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "disk_dir": str(disk_dir) if disk_dir is not None else None,
        "dual_write_events": len(events),
        "base_locations": base_locations,
        "full_locations": full_locations,
        "disk_file_count": len(disk_files),
        "full_disk_file_count": len(full_disk_files),
        "base_disk_file_count": len(base_disk_files),
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
