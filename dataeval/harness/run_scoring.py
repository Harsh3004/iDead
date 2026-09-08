"""CLI entry point for Step 8: Naive dead-reckoning baseline evaluation and leaderboard scoring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from dataeval.harness.baseline import (
    DEFAULT_PRE_OUTAGE_WINDOW_S,
    extract_pre_outage_state,
    propagate_constant_velocity_heading,
)
from dataeval.harness.metrics import OutageMetrics, compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory


CONFIG_NAME = "baseline_cv_heading_v1"

LEADERBOARD_COLUMNS = [
    # Locked schema (11 columns from Step 1)
    "run_id",
    "timestamp",
    "config",
    "outage_s",
    "final_pos_error_m",
    "pct_of_distance",
    "cep50_m",
    "cep95_m",
    "along_track_m",
    "cross_track_m",
    "heading_error_deg",
    # Extra trace-back columns
    "outage_id",
    "split",
    "source_side",
    "has_ground_truth",
    "skip_reason",
]


def score_outage_dataset(
    outages_dir: Path = Path("data/processed/outages"),
    leaderboard_csv: Path = Path("results/leaderboard.csv"),
    config_name: str = CONFIG_NAME,
    pre_outage_window_s: float = DEFAULT_PRE_OUTAGE_WINDOW_S,
) -> int:
    """Execute baseline propagation and score all valid outage instances against ground truth.

    Args:
        outages_dir: Path to directory containing outages manifest and split subdirectories.
        leaderboard_csv: Destination path for results/leaderboard.csv.
        config_name: Model/baseline configuration identifier.
        pre_outage_window_s: Pre-outage lookback window to estimate speed and heading.

    Returns:
        Exit code: 0 on success.
    """
    start_time = time.time()
    manifest_path = outages_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"Error: Outage manifest not found at {manifest_path}")
        return 1

    manifest = pd.read_csv(manifest_path)
    ok_instances = manifest[manifest["status"] == "ok"].copy()
    if ok_instances.empty:
        print("No valid outage instances (status == 'ok') found in manifest.")
        return 1

    print("Starting Dead-Reckoning Baseline Scoring:")
    print(f"  Outages directory:   {outages_dir.resolve()}")
    print(f"  Leaderboard path:    {leaderboard_csv.resolve()}")
    print(f"  Baseline config:     {config_name}")
    print(f"  Pre-outage window:   {pre_outage_window_s:.1f}s")
    print(f"  Candidate instances: {len(ok_instances)} total ({len(manifest)} manifest rows)\n")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    leaderboard_rows: List[Dict[str, any]] = []

    scored_gt_count = 0
    scored_nogt_count = 0
    gt_coverage_gap_count = 0

    for _, row in ok_instances.iterrows():
        outage_id = str(row["outage_id"]).strip()
        run_or_pair_id = str(row["run_or_pair_id"]).strip()
        split = str(row["split"]).strip()
        source_side = str(row["source_side"]).strip()
        outage_s = int(row["outage_s"]) if float(row["outage_s"]).is_integer() else float(row["outage_s"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        has_gt = bool(row["has_cross_stream_ground_truth"])

        parquet_path = outages_dir / split / f"{outage_id}.parquet"
        if not parquet_path.exists():
            print(f"Warning: Parquet file missing: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        window = OutageWindow(
            run_id=run_or_pair_id,
            source_side=source_side,
            start_s=start_s,
            duration_s=float(outage_s),
            end_s=end_s,
        )

        # 1. Extract pre-outage state (strictly t < start_s)
        pre_state = extract_pre_outage_state(
            df=df,
            window=window,
            pre_outage_window_s=pre_outage_window_s,
        )

        if pre_state is None:
            # Insufficient pre-outage GNSS fixes
            leaderboard_rows.append({
                "run_id": run_or_pair_id,
                "timestamp": run_timestamp,
                "config": config_name,
                "outage_s": outage_s,
                "final_pos_error_m": np.nan,
                "pct_of_distance": np.nan,
                "cep50_m": np.nan,
                "cep95_m": np.nan,
                "along_track_m": np.nan,
                "cross_track_m": np.nan,
                "heading_error_deg": np.nan,
                "outage_id": outage_id,
                "split": split,
                "source_side": source_side,
                "has_ground_truth": has_gt,
                "skip_reason": "insufficient_pre_outage_data: could not establish initial state",
            })
            continue

        # 2. Target timestamps within the outage window
        active_mask = (df["timestamp_s"] >= start_s) & (df["timestamp_s"] < end_s)
        target_timestamps = df.loc[active_mask, "timestamp_s"].values

        # 3. Propagate baseline trajectory forward
        pred_df = propagate_constant_velocity_heading(pre_state, target_timestamps)

        # 4. Score against ground truth if available
        if has_gt:
            gt_df = ground_truth_trajectory(df, window)
            if gt_df is None or len(gt_df) < 2 or gt_df["latitude_deg"].isna().any():
                gt_coverage_gap_count += 1
                metrics = OutageMetrics(
                    status="insufficient_ground_truth",
                    skip_reason="insufficient_ground_truth_coverage: opposite stream has gaps or excessive nulls",
                )
            else:
                metrics = compute_outage_metrics(pred_df, gt_df)

            leaderboard_rows.append({
                "run_id": run_or_pair_id,
                "timestamp": run_timestamp,
                "config": config_name,
                "outage_s": outage_s,
                "final_pos_error_m": metrics.final_pos_error_m,
                "pct_of_distance": metrics.pct_of_distance,
                "cep50_m": metrics.cep50_m,
                "cep95_m": metrics.cep95_m,
                "along_track_m": metrics.along_track_m,
                "cross_track_m": metrics.cross_track_m,
                "heading_error_deg": metrics.heading_error_deg,
                "outage_id": outage_id,
                "split": split,
                "source_side": source_side,
                "has_ground_truth": True,
                "skip_reason": metrics.skip_reason,
            })
            scored_gt_count += 1
        else:
            # Unpaired single-stream run: no independent ground truth during outage
            leaderboard_rows.append({
                "run_id": run_or_pair_id,
                "timestamp": run_timestamp,
                "config": config_name,
                "outage_s": outage_s,
                "final_pos_error_m": np.nan,
                "pct_of_distance": np.nan,
                "cep50_m": np.nan,
                "cep95_m": np.nan,
                "along_track_m": np.nan,
                "cross_track_m": np.nan,
                "heading_error_deg": np.nan,
                "outage_id": outage_id,
                "split": split,
                "source_side": source_side,
                "has_ground_truth": False,
                "skip_reason": "unpaired_run: no independent cross-stream ground truth",
            })
            scored_nogt_count += 1

    new_df = pd.DataFrame(leaderboard_rows)[LEADERBOARD_COLUMNS]

    # Non-destructive append to leaderboard.csv
    leaderboard_csv.parent.mkdir(parents=True, exist_ok=True)
    if leaderboard_csv.exists() and leaderboard_csv.stat().st_size > 0:
        # Check if existing file has data rows or only header
        existing_df = pd.read_csv(leaderboard_csv)
        if existing_df.empty:
            # Overwrite empty header with full 16-column header and new rows
            new_df.to_csv(leaderboard_csv, index=False)
        else:
            # Align columns and append without header
            new_df.to_csv(leaderboard_csv, mode="a", header=False, index=False)
    else:
        new_df.to_csv(leaderboard_csv, index=False)

    elapsed = time.time() - start_time

    # Compute headline summary metrics for the GT-verified subset
    gt_df_scored = new_df[new_df["has_ground_truth"] == True]
    summary_parts: List[str] = []
    durations = [10, 30, 60, 120, 180]

    for d in durations:
        sub = gt_df_scored[gt_df_scored["outage_s"] == d]
        if sub.empty:
            continue
        med_pos = sub["final_pos_error_m"].median()
        p95_pos = sub["final_pos_error_m"].quantile(0.95)
        med_pct = sub["pct_of_distance"].median()
        p95_pct = sub["pct_of_distance"].quantile(0.95)
        summary_parts.append(
            f"{d}s (N={len(sub)}): pos_err med={med_pos:.1f}m, p95={p95_pos:.1f}m | %dist med={med_pct:.1f}%, p95={p95_pct:.1f}%"
        )

    print("=" * 80)
    print(
        f"SCORING COMPLETE in {elapsed:.2f}s | "
        f"Total Scored: {len(new_df)} ({scored_gt_count} with GT, {scored_nogt_count} no-GT) | "
        f"GT Gaps Encountered: {gt_coverage_gap_count} | "
        f"Leaderboard: {leaderboard_csv}"
    )
    print("Baseline Performance (Ground-Truth-Verified Subset):")
    for sp in summary_parts:
        print(f"  • {sp}")
    print("=" * 80)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score dead-reckoning baseline models against ground-truth outages."
    )
    parser.add_argument(
        "--outages-dir",
        type=Path,
        default=Path("data/processed/outages"),
        help="Root directory containing outage parquets and manifest (default: data/processed/outages).",
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=Path("results/leaderboard.csv"),
        help="Path to leaderboard CSV file (default: results/leaderboard.csv).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_NAME,
        help=f"Baseline configuration name (default: {CONFIG_NAME}).",
    )
    parser.add_argument(
        "--pre-outage-window-s",
        type=float,
        default=DEFAULT_PRE_OUTAGE_WINDOW_S,
        help=f"Lookback window in seconds to estimate pre-outage state (default: {DEFAULT_PRE_OUTAGE_WINDOW_S}s).",
    )

    args = parser.parse_args()

    return score_outage_dataset(
        outages_dir=args.outages_dir,
        leaderboard_csv=args.leaderboard_csv,
        config_name=args.config,
        pre_outage_window_s=args.pre_outage_window_s,
    )


if __name__ == "__main__":
    sys.exit(main())
