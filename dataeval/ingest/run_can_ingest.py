#!/usr/bin/env python3
"""CLI entry point to execute vehicle/CAN dataset ingestion and Parquet conversion."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dataeval.ingest.vehicle_can import ingest_all_can_runs


def find_repo_root() -> Path:
    """Find repository root by looking for CMakeLists.txt and data/ directory."""
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "CMakeLists.txt").exists() and (parent / "data").exists():
            return parent
    return Path.cwd()


def main() -> int:
    repo_root = find_repo_root()
    default_raw = repo_root / "data" / "raw" / "io-vnbd" / "IO-VNBD"
    default_out = repo_root / "data" / "processed" / "vehicle_can"

    parser = argparse.ArgumentParser(description="Ingest and deduplicate vehicle/CAN CSV dataset into Parquet.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_raw,
        help=f"Path to raw dataset root (default: {default_raw})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"Destination directory for Parquet files and manifest (default: {default_out})",
    )
    args = parser.parse_args()

    print(f"Starting CAN Ingestion:")
    print(f"  Raw root: {args.raw_root}")
    print(f"  Out dir:  {args.out_dir}")

    start_t = time.perf_counter()
    entries = ingest_all_can_runs(args.raw_root, args.out_dir)
    elapsed_s = time.perf_counter() - start_t

    ok_entries = [e for e in entries if e.parse_status == "ok"]
    failed_entries = [e for e in entries if e.parse_status == "failed"]
    total_rows = sum(e.n_rows for e in ok_entries)

    # Compute total size of generated parquet files
    parquet_files = list(args.out_dir.glob("*.parquet"))
    total_size_bytes = sum(f.stat().st_size for f in parquet_files)
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"\nIngestion Complete in {elapsed_s:.2f}s:")
    print(
        f"Summary: runs ok: {len(ok_entries)} / failed: {len(failed_entries)} / "
        f"total rows written: {total_rows:,} / total output size: {total_size_mb:.2f} MB"
    )

    if failed_entries:
        print("\nFailed runs:")
        for f_entry in failed_entries:
            print(f"  - {f_entry.run_id}: {f_entry.parse_error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
