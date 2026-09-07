"""CLI entry point for Step 5: Paired CAN and Smartphone synchronization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List
import numpy as np
import pandas as pd

from dataeval.harness.pairing import PairEntry, discover_pairs
from dataeval.harness.sync import (
    AlignedPairResult,
    OffsetResult,
    align_pair,
    estimate_offset,
    write_paired_parquet,
)


def run_paired_sync(
    processed_root: Path,
    out_dir: Path,
    max_lag_s: float = 30.0,
) -> int:
    """Execute paired synchronization and Parquet serialization for all discovered pairs.

    Args:
        processed_root: Path to data/processed containing vehicle_can and smartphone directories.
        out_dir: Path to output directory (e.g. data/processed/paired).
        max_lag_s: Maximum lag search window in seconds.

    Returns:
        Exit code: 0 on success, 1 on critical failure.
    """
    start_time = time.time()
    can_manifest_path = processed_root / "vehicle_can" / "manifest.csv"
    phone_manifest_path = processed_root / "smartphone" / "manifest.csv"

    print("Starting Paired CAN <-> Smartphone Synchronization:")
    print(f"  Processed root: {processed_root.resolve()}")
    print(f"  Output dir:     {out_dir.resolve()}")

    pairs = discover_pairs(can_manifest_path, phone_manifest_path, processed_root)
    print(f"Discovered {len(pairs)} candidate pairs.")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    n_ok = 0
    n_failed = 0
    n_skipped = 0
    total_aligned_rows = 0
    confidences: List[float] = []

    for entry in pairs:
        pair_id = entry.pair_id
        flags: List[str] = list(entry.source_quality_flags)

        if entry.status != "ready":
            n_skipped += 1
            manifest_rows.append(
                {
                    "pair_id": pair_id,
                    "can_run_id": entry.can_run_id,
                    "phone_run_id": entry.phone_run_id,
                    "estimated_offset_s": 0.0,
                    "offset_confidence": 0.0,
                    "aligned_n_rows": 0,
                    "aligned_duration_s": 0.0,
                    "overlap_pct_of_can": 0.0,
                    "overlap_pct_of_phone": 0.0,
                    "quality_flags": "|".join(flags) if flags else "none",
                    "align_status": "skipped_missing_side",
                    "align_error": entry.status_detail,
                }
            )
            continue

        try:
            can_df = pd.read_parquet(entry.can_parquet_path)
            phone_df = pd.read_parquet(entry.phone_parquet_path)

            offset_res: OffsetResult = estimate_offset(can_df, phone_df, max_lag_s=max_lag_s)
            for f in offset_res.quality_flags:
                if f not in flags:
                    flags.append(f)

            aligned_res: AlignedPairResult = align_pair(
                can_df, phone_df, offset_s=offset_res.estimated_offset_s
            )

            write_paired_parquet(aligned_res.aligned_df, pair_id, out_dir)

            n_ok += 1
            total_aligned_rows += aligned_res.aligned_n_rows
            confidences.append(offset_res.offset_confidence)

            manifest_rows.append(
                {
                    "pair_id": pair_id,
                    "can_run_id": entry.can_run_id,
                    "phone_run_id": entry.phone_run_id,
                    "estimated_offset_s": round(offset_res.estimated_offset_s, 3),
                    "offset_confidence": round(offset_res.offset_confidence, 4),
                    "aligned_n_rows": aligned_res.aligned_n_rows,
                    "aligned_duration_s": round(aligned_res.aligned_duration_s, 3),
                    "overlap_pct_of_can": aligned_res.overlap_pct_of_can,
                    "overlap_pct_of_phone": aligned_res.overlap_pct_of_phone,
                    "quality_flags": "|".join(flags) if flags else "none",
                    "align_status": "ok",
                    "align_error": "",
                }
            )

        except Exception as ex:
            n_failed += 1
            manifest_rows.append(
                {
                    "pair_id": pair_id,
                    "can_run_id": entry.can_run_id,
                    "phone_run_id": entry.phone_run_id,
                    "estimated_offset_s": 0.0,
                    "offset_confidence": 0.0,
                    "aligned_n_rows": 0,
                    "aligned_duration_s": 0.0,
                    "overlap_pct_of_can": 0.0,
                    "overlap_pct_of_phone": 0.0,
                    "quality_flags": "|".join(flags) if flags else "none",
                    "align_status": "failed",
                    "align_error": str(ex),
                }
            )

    # Write manifest
    df_manifest = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)

    elapsed = time.time() - start_time
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    min_conf = float(np.min(confidences)) if confidences else 0.0

    print(f"\nPaired Ingestion Complete in {elapsed:.2f}s:")
    print(
        f"Summary: pairs ok: {n_ok} / failed: {n_failed} / skipped: {n_skipped} / "
        f"total aligned rows: {total_aligned_rows:,} / "
        f"mean confidence: {mean_conf:.3f} (min: {min_conf:.3f})"
    )

    return 0 if n_failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronise paired CAN and Smartphone datasets.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing processed vehicle_can and smartphone datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/paired"),
        help="Output directory for aligned Parquet files and manifest.",
    )
    parser.add_argument(
        "--max-lag-s",
        type=float,
        default=30.0,
        help="Maximum lag search window for offset cross-correlation (seconds).",
    )
    args = parser.parse_args()

    return run_paired_sync(
        processed_root=args.processed_root,
        out_dir=args.out_dir,
        max_lag_s=args.max_lag_s,
    )


if __name__ == "__main__":
    sys.exit(main())
