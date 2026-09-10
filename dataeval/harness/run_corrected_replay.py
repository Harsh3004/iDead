"""Orchestration script for Step 16: Scoring and evaluation of attitude-corrected C++ strapdown replay.

Loads predictions from data/processed/cpp_predictions_corrected/, evaluates against
independent ground truth, appends 253 rows to results/leaderboard.csv under
config='corrected_init_strapdown_v1', and generates tiered comparison reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.metrics import OutageMetrics, compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory


CONFIG_CV = "baseline_cv_heading_v1"
CONFIG_CPP_BASE = "baseline_cpp_strapdown_v1"
CONFIG_CORRECTED = "corrected_init_strapdown_v1"

LEADERBOARD_COLUMNS = [
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
    "outage_id",
    "split",
    "source_side",
    "has_ground_truth",
    "skip_reason",
]


def load_module_b_tiers(
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
) -> Dict[str, Dict[str, Any]]:
    """Load run-level tier classification from Module B attitude synthesis results."""
    if not attitude_csv.exists():
        raise FileNotFoundError(f"Attitude CSV not found at {attitude_csv}")

    df = pd.read_csv(attitude_csv, comment="#")
    tiers: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        run_id = str(row["run_id"])
        has_motion = bool(row["has_motion"]) if pd.notna(row["has_motion"]) else False
        leveling_source = str(row["leveling_source"]) if pd.notna(row["leveling_source"]) else "fallback_zero"
        gyro_bias_source = str(row["gyro_bias_source"]) if pd.notna(row["gyro_bias_source"]) else "none"

        if leveling_source == "module_a_measured":
            tier = "module_a_measured"
        else:
            tier = "fallback_zero"

        tiers[run_id] = {
            "tier": tier,
            "has_motion": has_motion,
            "leveling_source": leveling_source,
            "gyro_bias_source": gyro_bias_source,
            "flag": str(row.get("flag", "")),
        }
    return tiers


def run_cpp_batch_replay(
    binary_path: Path = Path("build/core/idr_replay.exe"),
    cache_dir: Path = Path("data/processed/_cpp_replay_cache"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
    out_dir: Path = Path("data/processed/cpp_predictions_corrected"),
) -> None:
    """Execute C++ replay driver in batch mode with attitude injection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary_path),
        "--cache-dir", str(cache_dir),
        "--attitude-csv", str(attitude_csv),
        "--out-dir", str(out_dir),
        "--batch",
    ]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"C++ replay driver failed with exit code {res.returncode}")


def score_corrected_predictions(
    manifest_csv: Path = Path("data/processed/outages/manifest.csv"),
    outages_dir: Path = Path("data/processed/outages"),
    predictions_dir: Path = Path("data/processed/cpp_predictions_corrected"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
    leaderboard_csv: Path = Path("results/leaderboard.csv"),
) -> pd.DataFrame:
    """Score all 253 eligible paired outage instances and append to results/leaderboard.csv."""
    tiers = load_module_b_tiers(attitude_csv)
    eligible_run_ids = set(tiers.keys())

    manifest = pd.read_csv(manifest_csv)
    # Filter to eligible paired instances (status == 'ok' and has_cross_stream_ground_truth == True)
    eligible_mask = (
        (manifest["status"] == "ok")
        & (manifest["has_cross_stream_ground_truth"] == True)
        & (manifest["run_or_pair_id"].isin(eligible_run_ids))
    )
    eligible_manifest = manifest[eligible_mask].copy()

    print(f"Total manifest instances: {len(manifest)}")
    print(f"Eligible paired instances (matching 72 paired runs): {len(eligible_manifest)}")

    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    scored_rows: List[Dict[str, Any]] = []

    for _, row in eligible_manifest.iterrows():
        outage_id = str(row["outage_id"]).strip()
        run_or_pair_id = str(row["run_or_pair_id"]).strip()
        split = str(row["split"]).strip()
        outage_s = int(row["outage_s"]) if float(row["outage_s"]).is_integer() else float(row["outage_s"])
        source_side = str(row["source_side"]).strip()
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])

        pred_csv = predictions_dir / f"{outage_id}.csv"
        if not pred_csv.exists():
            raise FileNotFoundError(f"Missing prediction file for eligible outage: {pred_csv}")

        pred_df = load_cpp_prediction(pred_csv)

        outage_parquet = outages_dir / split / f"{outage_id}.parquet"
        if not outage_parquet.exists():
            raise FileNotFoundError(f"Missing outage parquet file: {outage_parquet}")

        df = pd.read_parquet(outage_parquet)
        window = OutageWindow(
            run_id=run_or_pair_id,
            source_side=source_side,
            start_s=start_s,
            duration_s=float(outage_s),
            end_s=end_s,
        )

        gt_df = ground_truth_trajectory(df, window)
        if gt_df is None or len(gt_df) < 2 or gt_df["latitude_deg"].isna().any():
            metrics = OutageMetrics(
                status="insufficient_ground_truth",
                skip_reason="insufficient_ground_truth_coverage: opposite stream has gaps or excessive nulls",
            )
        else:
            metrics = compute_outage_metrics(pred_df, gt_df)

        scored_rows.append({
            "run_id": run_or_pair_id,
            "timestamp": run_timestamp,
            "config": CONFIG_CORRECTED,
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

    new_df = pd.DataFrame(scored_rows)[LEADERBOARD_COLUMNS]

    # Verify existing leaderboard before appending
    if leaderboard_csv.exists():
        existing_df = pd.read_csv(leaderboard_csv)
        initial_count = len(existing_df)
        print(f"Existing leaderboard rows before append: {initial_count}")
        # Append without header to preserve existing records identically
        new_df.to_csv(leaderboard_csv, mode="a", header=False, index=False)
        verified_df = pd.read_csv(leaderboard_csv)
        print(f"Leaderboard rows after append: {len(verified_df)} (+{len(new_df)})")
        assert len(verified_df) == initial_count + len(new_df), "Row count mismatch after append!"
    else:
        new_df.to_csv(leaderboard_csv, index=False)

    elapsed = time.time() - start_time
    print(f"Scoring completed in {elapsed:.2f}s for {len(new_df)} instances.")
    return new_df


def compute_tiered_comparison_report(
    leaderboard_csv: Path = Path("results/leaderboard.csv"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
) -> str:
    """Produce comprehensive Tiered Comparison Report across durations and sub-populations."""
    df = pd.read_csv(leaderboard_csv)
    tiers = load_module_b_tiers(attitude_csv)

    # Attach tier info
    df["tier"] = df["run_id"].map(lambda r: tiers.get(r, {}).get("tier", "unknown"))

    durations = [10, 30, 60, 120, 180]
    configs = [CONFIG_CV, CONFIG_CPP_BASE, CONFIG_CORRECTED]

    report_lines: List[str] = []
    report_lines.append("# Step 16: Initial Attitude & Gyro Bias Correction Tiered Report\n")

    # 1. Overall comparison on all 253 eligible instances
    report_lines.append("## 1. All Eligible Paired Instances (N=253)\n")
    report_lines.append("| Outage | N | Baseline CV Med (m) | Bare Strapdown Med (m) | Corrected Strapdown Med (m) | Bare p95 (m) | Corrected p95 (m) | Improvement vs Bare |")
    report_lines.append("|---|---|---|---|---|---|---|---|")

    eligible_df = df[df["run_id"].isin(tiers.keys())]

    for d in durations:
        sub_cv = eligible_df[(eligible_df["config"] == CONFIG_CV) & (eligible_df["outage_s"] == d)]
        sub_bare = eligible_df[(eligible_df["config"] == CONFIG_CPP_BASE) & (eligible_df["outage_s"] == d)]
        sub_corr = eligible_df[(eligible_df["config"] == CONFIG_CORRECTED) & (eligible_df["outage_s"] == d)]

        n = len(sub_corr)
        med_cv = sub_cv["final_pos_error_m"].median()
        med_bare = sub_bare["final_pos_error_m"].median()
        med_corr = sub_corr["final_pos_error_m"].median()
        p95_bare = sub_bare["final_pos_error_m"].quantile(0.95)
        p95_corr = sub_corr["final_pos_error_m"].quantile(0.95)

        imp = ((med_bare - med_corr) / med_bare) * 100.0 if med_bare > 0 else 0.0
        report_lines.append(
            f"| {d}s | {n} | {med_cv:.1f} | {med_bare:.1f} | {med_corr:.1f} | {p95_bare:.1f} | {p95_corr:.1f} | {imp:+.1f}% |"
        )

    # 2. Calibrated Tier only (module_a_measured: 155 instances)
    report_lines.append("\n## 2. Calibrated Tier: `module_a_measured` (N=155 instances)\n")
    report_lines.append("Runs with measured stationary pitch/roll leveling AND measured gyro bias from stillness.\n")
    report_lines.append("| Outage | N | Bare Strapdown Med (m) | Corrected Strapdown Med (m) | Delta Med (m) | Improvement | Bare p95 (m) | Corrected p95 (m) |")
    report_lines.append("|---|---|---|---|---|---|---|---|")

    calib_df = eligible_df[eligible_df["tier"] == "module_a_measured"]
    for d in durations:
        sub_bare = calib_df[(calib_df["config"] == CONFIG_CPP_BASE) & (calib_df["outage_s"] == d)]
        sub_corr = calib_df[(calib_df["config"] == CONFIG_CORRECTED) & (calib_df["outage_s"] == d)]

        n = len(sub_corr)
        med_bare = sub_bare["final_pos_error_m"].median()
        med_corr = sub_corr["final_pos_error_m"].median()
        p95_bare = sub_bare["final_pos_error_m"].quantile(0.95)
        p95_corr = sub_corr["final_pos_error_m"].quantile(0.95)
        delta = med_corr - med_bare
        imp = ((med_bare - med_corr) / med_bare) * 100.0 if med_bare > 0 else 0.0

        report_lines.append(
            f"| {d}s | {n} | {med_bare:.1f} | {med_corr:.1f} | {delta:+.1f}m | {imp:+.1f}% | {p95_bare:.1f} | {p95_corr:.1f} |"
        )

    # 3. Fallback Tier only (fallback_zero: 98 instances)
    report_lines.append("\n## 3. Fallback Tier: `fallback_zero` (N=98 instances)\n")
    report_lines.append("Runs with PCA motion yaw but flat leveling and ZERO gyro bias correction.\n")
    report_lines.append("| Outage | N | Bare Strapdown Med (m) | Corrected Strapdown Med (m) | Delta Med (m) | Improvement | Bare p95 (m) | Corrected p95 (m) |")
    report_lines.append("|---|---|---|---|---|---|---|---|")

    fb_df = eligible_df[eligible_df["tier"] == "fallback_zero"]
    for d in durations:
        sub_bare = fb_df[(fb_df["config"] == CONFIG_CPP_BASE) & (fb_df["outage_s"] == d)]
        sub_corr = fb_df[(fb_df["config"] == CONFIG_CORRECTED) & (fb_df["outage_s"] == d)]

        n = len(sub_corr)
        med_bare = sub_bare["final_pos_error_m"].median()
        med_corr = sub_corr["final_pos_error_m"].median()
        p95_bare = sub_bare["final_pos_error_m"].quantile(0.95)
        p95_corr = sub_corr["final_pos_error_m"].quantile(0.95)
        delta = med_corr - med_bare
        imp = ((med_bare - med_corr) / med_bare) * 100.0 if med_bare > 0 else 0.0

        report_lines.append(
            f"| {d}s | {n} | {med_bare:.1f} | {med_corr:.1f} | {delta:+.1f}m | {imp:+.1f}% | {p95_bare:.1f} | {p95_corr:.1f} |"
        )

    # 4. Suspect Runs Callout: pair_S2 and pair_Vta17
    report_lines.append("\n## 4. Suspect Bias Runs Callout: `pair_S2` and `pair_Vta17`\n")
    report_lines.append("Cross-check in Step 15 flagged high gyro discrepancies (35-52 deg), indicating possible corrupted stillness windows.\n")
    report_lines.append("| Outage ID | Outage (s) | Bare Pos Error (m) | Corrected Pos Error (m) | Delta Error (m) | Outcome |")
    report_lines.append("|---|---|---|---|---|---|")

    suspect_runs = ["pair_S2", "pair_Vta17"]
    suspect_df = eligible_df[eligible_df["run_id"].isin(suspect_runs)]
    suspect_outages = suspect_df[suspect_df["config"] == CONFIG_CORRECTED]["outage_id"].unique()

    for oid in sorted(suspect_outages):
        row_bare = suspect_df[(suspect_df["config"] == CONFIG_CPP_BASE) & (suspect_df["outage_id"] == oid)]
        row_corr = suspect_df[(suspect_df["config"] == CONFIG_CORRECTED) & (suspect_df["outage_id"] == oid)]
        if row_bare.empty or row_corr.empty:
            continue
        err_bare = float(row_bare.iloc[0]["final_pos_error_m"])
        err_corr = float(row_corr.iloc[0]["final_pos_error_m"])
        d_s = int(row_corr.iloc[0]["outage_s"])
        delta = err_corr - err_bare
        outcome = "Improved" if delta < 0 else "Degraded"
        report_lines.append(
            f"| {oid} | {d_s}s | {err_bare:.1f} | {err_corr:.1f} | {delta:+.1f}m | {outcome} |"
        )

    return "\n".join(report_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score attitude-corrected strapdown replay.")
    parser.add_argument("--skip-replay", action="store_true", help="Skip C++ replay execution.")
    args = parser.parse_args()

    if not args.skip_replay:
        run_cpp_batch_replay()

    score_corrected_predictions()
    report = compute_tiered_comparison_report()
    print("\n" + report)
