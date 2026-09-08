"""CLI entry point for dead-reckoning baseline evaluation and leaderboard scoring.

Supports both:
1. Python naive constant-velocity baseline: config='baseline_cv_heading_v1'
2. C++ strapdown INS core baseline: config='baseline_cpp_strapdown_v1'
"""

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
from dataeval.harness.cpp_predictions import (
    load_cpp_prediction,
    spot_check_predictions,
)
from dataeval.harness.metrics import OutageMetrics, compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory


CONFIG_CV = "baseline_cv_heading_v1"
CONFIG_CPP = "baseline_cpp_strapdown_v1"

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


def generate_comparison_table(leaderboard_csv: Path) -> str:
    """Generate formatted side-by-side comparison table for paired instances across all outage lengths."""
    if not leaderboard_csv.exists() or leaderboard_csv.stat().st_size == 0:
        return ""

    df = pd.read_csv(leaderboard_csv)
    gt_df = df[df["has_ground_truth"] == True]

    cv_df = gt_df[gt_df["config"] == CONFIG_CV]
    cpp_df = gt_df[gt_df["config"] == CONFIG_CPP]

    if cv_df.empty or cpp_df.empty:
        return ""

    durations = [10, 30, 60, 120, 180]
    lines = [
        "| Outage | N | CV Median (m) | CV p95 (m) | Strapdown Med (m) | Strapdown p95 (m) | Delta Med (m) | CV %Dist Med | Strapdown %Dist Med |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for d in durations:
        sub_cv = cv_df[cv_df["outage_s"] == d]
        sub_cpp = cpp_df[cpp_df["outage_s"] == d]
        n_cv = len(sub_cv)
        n_cpp = len(sub_cpp)
        n = min(n_cv, n_cpp)
        if n == 0:
            continue

        med_cv = sub_cv["final_pos_error_m"].median()
        p95_cv = sub_cv["final_pos_error_m"].quantile(0.95)
        dist_cv = sub_cv["pct_of_distance"].median()

        med_cpp = sub_cpp["final_pos_error_m"].median()
        p95_cpp = sub_cpp["final_pos_error_m"].quantile(0.95)
        dist_cpp = sub_cpp["pct_of_distance"].median()

        delta_med = med_cpp - med_cv
        delta_str = f"{delta_med:+.1f}"

        dist_cv_str = f"{dist_cv:.1f}%" if pd.notna(dist_cv) else "N/A"
        dist_cpp_str = f"{dist_cpp:.1f}%" if pd.notna(dist_cpp) else "N/A"

        lines.append(
            f"| {d}s | {n} | {med_cv:.1f} | {p95_cv:.1f} | {med_cpp:.1f} | {p95_cpp:.1f} | {delta_str} | {dist_cv_str} | {dist_cpp_str} |"
        )

    return "\n".join(lines)


def score_outage_dataset(
    outages_dir: Path = Path("data/processed/outages"),
    leaderboard_csv: Path = Path("results/leaderboard.csv"),
    config_name: str = CONFIG_CV,
    prediction_dir: Optional[Path] = None,
    pre_outage_window_s: float = DEFAULT_PRE_OUTAGE_WINDOW_S,
    do_spot_check: bool = False,
) -> int:
    """Execute evaluation and score all valid outage instances against ground truth.

    Args:
        outages_dir: Path to directory containing outages manifest and split subdirectories.
        leaderboard_csv: Destination path for results/leaderboard.csv.
        config_name: Model/baseline configuration identifier.
        prediction_dir: Directory containing precomputed prediction CSVs (required for C++ baseline).
        pre_outage_window_s: Pre-outage lookback window to estimate speed and heading.
        do_spot_check: If True, performs spot-check validation on prediction files.

    Returns:
        Exit code: 0 on success.
    """
    start_time = time.time()
    manifest_path = outages_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"Error: Outage manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(manifest_path)
    ok_instances = manifest[manifest["status"] == "ok"].copy()
    if ok_instances.empty:
        print("No valid outage instances (status == 'ok') found in manifest.", file=sys.stderr)
        return 1

    is_cpp_mode = (config_name == CONFIG_CPP) or (prediction_dir is not None)
    if is_cpp_mode and prediction_dir is None:
        prediction_dir = Path("data/processed/cpp_predictions")

    print("Starting Dead-Reckoning Scoring Harness:")
    print(f"  Outages directory:   {outages_dir.resolve()}")
    print(f"  Leaderboard path:    {leaderboard_csv.resolve()}")
    print(f"  Model/Config:        {config_name}")
    if is_cpp_mode:
        print(f"  Predictions source:  {prediction_dir.resolve()}")
    else:
        print(f"  Pre-outage window:   {pre_outage_window_s:.1f}s")
    print(f"  Candidate instances: {len(ok_instances)} total ({len(manifest)} manifest rows)\n")

    # Run spot-checks if requested or if in C++ mode
    if is_cpp_mode and (do_spot_check or True):
        print("Running Pre-Scoring Spot-Check Validation on C++ Predictions:")
        spot_checks = spot_check_predictions(ok_instances, prediction_dir, sample_per_duration=2)
        valid_spot_count = sum(1 for sc in spot_checks if sc["valid"])
        print(f"  Checked {len(spot_checks)} stratified instances across 10-180s: {valid_spot_count}/{len(spot_checks)} valid")
        for sc in spot_checks:
            status_tag = "[OK]" if sc["valid"] else "[FAIL]"
            print(f"  {status_tag} {sc['outage_id']} ({sc['outage_s']}s, {sc['split']}): {sc.get('n_rows', 0)} rows, span={sc.get('span_s', 0):.2f}s")
            if not sc["valid"]:
                for iss in sc.get("issues", []):
                    print(f"       -> Issue: {iss}")
        print()

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

        pred_df: Optional[pd.DataFrame] = None
        pred_skip_reason: str = ""

        if is_cpp_mode:
            # Mode A: Load from precomputed C++ strapdown predictions
            pred_csv = prediction_dir / f"{outage_id}.csv"
            if not pred_csv.exists():
                pred_skip_reason = f"prediction_file_missing: {pred_csv.name} not found"
            else:
                try:
                    pred_df = load_cpp_prediction(pred_csv)
                except Exception as exc:
                    pred_skip_reason = f"prediction_load_error: {exc}"
        else:
            # Mode B: Python Constant-Velocity / Heading propagation
            pre_state = extract_pre_outage_state(
                df=df,
                window=window,
                pre_outage_window_s=pre_outage_window_s,
            )
            if pre_state is None:
                pred_skip_reason = "insufficient_pre_outage_data: could not establish initial state"
            else:
                active_mask = (df["timestamp_s"] >= start_s) & (df["timestamp_s"] < end_s)
                target_timestamps = df.loc[active_mask, "timestamp_s"].values
                pred_df = propagate_constant_velocity_heading(pre_state, target_timestamps)

        if pred_df is None or len(pred_df) < 2:
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
                "skip_reason": pred_skip_reason,
            })
            continue

        # Score against ground truth if available
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
            # Unpaired single-stream run: no independent cross-stream ground truth
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
        existing_df = pd.read_csv(leaderboard_csv)
        if existing_df.empty:
            new_df.to_csv(leaderboard_csv, index=False)
        else:
            # Append without header to preserve existing records identically
            new_df.to_csv(leaderboard_csv, mode="a", header=False, index=False)
    else:
        new_df.to_csv(leaderboard_csv, index=False)

    elapsed = time.time() - start_time

    # Compute headline summary metrics for the GT-verified subset of this run
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
    print(f"Performance for config '{config_name}' (Ground-Truth-Verified Subset):")
    for sp in summary_parts:
        print(f"  • {sp}")
    print("=" * 80)

    # Generate side-by-side comparison table if both configs are present
    comparison_tbl = generate_comparison_table(leaderboard_csv)
    if comparison_tbl:
        print("\nSide-by-Side Comparison: Naive CV Baseline vs C++ Strapdown Baseline (Paired N=253):")
        print(comparison_tbl)
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
        default=CONFIG_CV,
        help=f"Model/baseline configuration identifier (default: {CONFIG_CV}, options: {CONFIG_CV}, {CONFIG_CPP}).",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Directory containing prediction CSV files (default: data/processed/cpp_predictions for C++ baseline).",
    )
    parser.add_argument(
        "--pre-outage-window-s",
        type=float,
        default=DEFAULT_PRE_OUTAGE_WINDOW_S,
        help=f"Lookback window in seconds to estimate pre-outage state (default: {DEFAULT_PRE_OUTAGE_WINDOW_S}s).",
    )
    parser.add_argument(
        "--spot-check",
        action="store_true",
        help="Run spot-check validation on prediction files before scoring.",
    )

    args = parser.parse_args()

    return score_outage_dataset(
        outages_dir=args.outages_dir,
        leaderboard_csv=args.leaderboard_csv,
        config_name=args.config,
        prediction_dir=args.prediction_dir,
        pre_outage_window_s=args.pre_outage_window_s,
        do_spot_check=args.spot_check,
    )


if __name__ == "__main__":
    sys.exit(main())
