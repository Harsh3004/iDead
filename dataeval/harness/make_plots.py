"""Generate publication-ready comparison plots and summary tables for Step 11.

Generates:
1. results/leaderboard_summary.csv: Grouped aggregate statistics.
2. results/plots/error_vs_outage_duration.png: Log-scale error vs outage duration.
3. results/plots/error_ratio_bar.png: Bar chart of Strapdown ÷ CV median error ratio.
4. results/plots/sample_trajectories.png: ENU trajectory comparison on 2 representative instances.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataeval.harness.baseline import (
    extract_pre_outage_state,
    propagate_constant_velocity_heading,
)
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
from dataeval.harness.summary import compute_leaderboard_summary


EARTH_RADIUS_M = 6371000.0


def plot_error_vs_outage_duration(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot median and p95 position error vs outage duration on a log scale."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    cv_df = summary_df[summary_df["config"] == "baseline_cv_heading_v1"].sort_values("outage_s")
    cpp_df = summary_df[summary_df["config"] == "baseline_cpp_strapdown_v1"].sort_values("outage_s")

    x = cv_df["outage_s"].values

    # Plot CV baseline (blue)
    ax.plot(
        x,
        cv_df["median_final_pos_error_m"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Constant-Velocity (Median)",
    )
    ax.plot(
        x,
        cv_df["p95_final_pos_error_m"],
        color="#1f77b4",
        linestyle="--",
        marker="o",
        markerfacecolor="none",
        linewidth=1.6,
        markersize=6,
        label="Constant-Velocity (95th percentile)",
    )
    ax.fill_between(
        x,
        cv_df["median_final_pos_error_m"],
        cv_df["p95_final_pos_error_m"],
        color="#1f77b4",
        alpha=0.10,
    )

    # Plot Strapdown baseline (crimson)
    ax.plot(
        x,
        cpp_df["median_final_pos_error_m"],
        color="#d62728",
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="C++ Strapdown INS (Median)",
    )
    ax.plot(
        x,
        cpp_df["p95_final_pos_error_m"],
        color="#d62728",
        linestyle="--",
        marker="s",
        markerfacecolor="none",
        linewidth=1.6,
        markersize=6,
        label="C++ Strapdown INS (95th percentile)",
    )
    ax.fill_between(
        x,
        cpp_df["median_final_pos_error_m"],
        cpp_df["p95_final_pos_error_m"],
        color="#d62728",
        alpha=0.10,
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}s" for d in x], fontsize=11)
    ax.set_xlabel("GNSS Outage Duration (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Position Error (meters, log scale)", fontsize=12, fontweight="bold")

    ax.set_title(
        "Dead-Reckoning Position Error vs. Outage Duration\n"
        "Paired Ground-Truth Benchmark (N=253) [Logarithmic Y-Scale]",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # Annotate ratio at endpoints if available
    sub_10_cv = cv_df[cv_df["outage_s"] == 10]
    sub_10_cpp = cpp_df[cpp_df["outage_s"] == 10]
    if not sub_10_cv.empty and not sub_10_cpp.empty:
        med_10_cv = sub_10_cv["median_final_pos_error_m"].iloc[0]
        med_10_cpp = sub_10_cpp["median_final_pos_error_m"].iloc[0]
        ratio_10 = med_10_cpp / med_10_cv if med_10_cv > 0 else 1.0
        ax.annotate(
            f"10s: {ratio_10:.1f}×\n({med_10_cpp:.0f}m vs {med_10_cv:.0f}m)",
            xy=(10, med_10_cpp),
            xytext=(15, med_10_cpp * 1.5),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
            fontsize=8.5,
            fontweight="semibold",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f9f9f9", ec="#cccccc", lw=0.6),
        )

    sub_180_cv = cv_df[cv_df["outage_s"] == 180]
    sub_180_cpp = cpp_df[cpp_df["outage_s"] == 180]
    if not sub_180_cv.empty and not sub_180_cpp.empty:
        med_180_cv = sub_180_cv["median_final_pos_error_m"].iloc[0]
        med_180_cpp = sub_180_cpp["median_final_pos_error_m"].iloc[0]
        ratio_180 = med_180_cpp / med_180_cv if med_180_cv > 0 else 1.0
        ax.annotate(
            f"180s: {ratio_180:.1f}×\n({med_180_cpp / 1000:.1f} km vs {med_180_cv / 1000:.1f} km)",
            xy=(180, med_180_cpp),
            xytext=(130, med_180_cpp * 0.9),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
            fontsize=8.5,
            fontweight="semibold",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f9f9f9", ec="#cccccc", lw=0.6),
        )

    ax.grid(True, which="both", linestyle=":", alpha=0.5, color="#888888")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9.5, loc="upper left")

    plt.tight_layout(pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_error_ratio_bar(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot bar chart of Strapdown median ÷ CV median error ratio per outage duration."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)

    cv_df = summary_df[summary_df["config"] == "baseline_cv_heading_v1"].set_index("outage_s")
    cpp_df = summary_df[summary_df["config"] == "baseline_cpp_strapdown_v1"].set_index("outage_s")

    available_durations = [d for d in [10, 30, 60, 120, 180] if d in cv_df.index and d in cpp_df.index]
    if not available_durations:
        plt.close(fig)
        return

    ratios = []
    labels = []

    for d in available_durations:
        cv_med = cv_df.loc[d, "median_final_pos_error_m"]
        cpp_med = cpp_df.loc[d, "median_final_pos_error_m"]
        ratio = cpp_med / cv_med if cv_med > 0 else 1.0
        ratios.append(ratio)
        labels.append(f"{d}s\n(N={int(cv_df.loc[d, 'n_paired'])})")

    x = np.arange(len(available_durations))
    base_colors = ["#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"]
    colors = base_colors[:len(available_durations)]

    bars = ax.bar(x, ratios, width=0.55, color=colors, edgecolor="#444444", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Error Ratio (Strapdown Median ÷ CV Median)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Error Divergence Ratio: Uncorrected Strapdown INS ÷ Constant-Velocity\n"
        "Widening Gap from Cubic Sensor Drift across Outage Durations (Paired N=253)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    ax.set_ylim(0, max(ratios) * 1.15)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7, label="1.0× Parity Line")

    # Add numeric callouts on each bar
    for bar, ratio, d in zip(bars, ratios, available_durations):
        height = bar.get_height()
        cv_val = cv_df.loc[d, "median_final_pos_error_m"]
        cpp_val = cpp_df.loc[d, "median_final_pos_error_m"]

        val_str = f"{cpp_val:.0f}m / {cv_val:.0f}m" if cpp_val < 1000 else f"{cpp_val / 1000:.1f}k / {cv_val:.0f}m"
        ax.annotate(
            f"{ratio:.1f}×\n({val_str})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def geodetic_to_enu(
    lats: np.ndarray,
    lons: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert geodetic latitude/longitude to local tangent plane ENU meters."""
    cos_ref = np.cos(np.deg2rad(ref_lat))
    d2r = np.pi / 180.0
    east = (lons - ref_lon) * d2r * EARTH_RADIUS_M * cos_ref
    north = (lats - ref_lat) * d2r * EARTH_RADIUS_M
    return east, north


def plot_sample_trajectories(
    outages_dir: Path,
    prediction_dir: Path,
    instance_short: str,
    instance_long: str,
    output_path: Path,
) -> None:
    """Plot ground truth vs CV vs Strapdown trajectories for 2 representative instances in local ENU meters."""
    manifest = pd.read_csv(outages_dir / "manifest.csv")
    instances = [instance_short, instance_long]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), dpi=300)

    for idx, (ax, oid) in enumerate(zip(axes, instances)):
        matching = manifest[manifest["outage_id"] == oid]
        if matching.empty:
            ax.set_title(f"Instance {oid} not found in manifest", fontsize=10)
            continue
        row = matching.iloc[0]
        parquet_p = outages_dir / row["split"] / f"{oid}.parquet"
        pred_p = prediction_dir / f"{oid}.csv"
        if not parquet_p.exists() or not pred_p.exists():
            ax.set_title(f"Files for {oid} not found on disk", fontsize=10)
            continue

        df = pd.read_parquet(parquet_p)

        window = OutageWindow(
            run_id=row["run_or_pair_id"],
            source_side=row["source_side"],
            start_s=float(row["start_s"]),
            duration_s=float(row["outage_s"]),
            end_s=float(row["end_s"]),
        )

        gt_df = ground_truth_trajectory(df, window)
        if gt_df is None or len(gt_df) < 2:
            ax.set_title(f"No ground truth trajectory for {oid}", fontsize=10)
            continue

        cpp_df = load_cpp_prediction(pred_p)
        pre_state = extract_pre_outage_state(df, window)
        cv_df = propagate_constant_velocity_heading(pre_state, gt_df["timestamp_s"].values)

        # Coordinate origin at ground-truth start
        lat0 = float(gt_df["latitude_deg"].iloc[0])
        lon0 = float(gt_df["longitude_deg"].iloc[0])

        gt_e, gt_n = geodetic_to_enu(gt_df["latitude_deg"].values, gt_df["longitude_deg"].values, lat0, lon0)
        cv_e, cv_n = geodetic_to_enu(cv_df["latitude_deg"].values, cv_df["longitude_deg"].values, lat0, lon0)
        cpp_e, cpp_n = geodetic_to_enu(cpp_df["latitude_deg"].values, cpp_df["longitude_deg"].values, lat0, lon0)

        # Compute endpoint errors
        cv_err = np.sqrt((cv_e[-1] - gt_e[-1]) ** 2 + (cv_n[-1] - gt_n[-1]) ** 2)
        cpp_err = np.sqrt((cpp_e[-1] - gt_e[-1]) ** 2 + (cpp_n[-1] - gt_n[-1]) ** 2)

        # Plot trajectories
        ax.plot(gt_e, gt_n, color="#2ca02c", linewidth=2.6, label="Ground Truth (Reference)", zorder=4)
        ax.plot(cv_e, cv_n, color="#1f77b4", linestyle="--", linewidth=2.0, label="Constant-Velocity Baseline", zorder=3)
        ax.plot(cpp_e, cpp_n, color="#d62728", linestyle="-.", linewidth=1.8, label="C++ Strapdown INS", zorder=2)

        # Markers
        ax.scatter([0], [0], color="#000000", marker="o", s=60, zorder=5, label="Outage Start Point")
        ax.scatter([gt_e[-1]], [gt_n[-1]], color="#2ca02c", marker="*", s=110, zorder=5, label="GT End")
        ax.scatter([cv_e[-1]], [cv_n[-1]], color="#1f77b4", marker="^", s=70, zorder=5, label="CV End")
        ax.scatter([cpp_e[-1]], [cpp_n[-1]], color="#d62728", marker="s", s=60, zorder=5, label="Strapdown End")

        outage_s = int(row["outage_s"])
        ax.set_title(
            f"Outage Instance: {oid} ({outage_s}s Outage, {row['split'].upper()})\n"
            f"End Error: CV = {cv_err:.1f}m | Strapdown = {cpp_err:.1f}m ({cpp_err / cv_err:.1f}×)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("East Displacement (meters)", fontsize=10, fontweight="bold")
        ax.set_ylabel("North Displacement (meters)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Equal aspect or balanced view
        ax.set_aspect("equal", adjustable="datalim")

        if idx == 0:
            ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8.5, loc="best")

    fig.suptitle(
        "Representative Trajectory Replay Comparison in Local Tangent Plane (ENU)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_all_plots(
    leaderboard_csv: Path = Path("results/leaderboard.csv"),
    summary_csv: Path = Path("results/leaderboard_summary.csv"),
    outages_dir: Path = Path("data/processed/outages"),
    prediction_dir: Path = Path("data/processed/cpp_predictions"),
    plots_dir: Path = Path("results/plots"),
    instance_short: str = "pair_Vfa01__10s__0",
    instance_long: str = "pair_Y1__180s__0",
) -> None:
    """Generate summary CSV and all three Step 11 plots."""
    print("Computing Leaderboard Summary CSV...")
    summary_df = compute_leaderboard_summary(leaderboard_csv, output_path=summary_csv)
    print(f"  [OK] Written: {summary_csv.resolve()}")

    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Plot 1: error_vs_outage_duration.png...")
    plot1_path = plots_dir / "error_vs_outage_duration.png"
    plot_error_vs_outage_duration(summary_df, plot1_path)
    print(f"  [OK] Saved: {plot1_path.resolve()}")

    print("Generating Plot 2: error_ratio_bar.png...")
    plot2_path = plots_dir / "error_ratio_bar.png"
    plot_error_ratio_bar(summary_df, plot2_path)
    print(f"  [OK] Saved: {plot2_path.resolve()}")

    print("Generating Plot 3: sample_trajectories.png...")
    plot3_path = plots_dir / "sample_trajectories.png"
    plot_sample_trajectories(outages_dir, prediction_dir, instance_short, instance_long, plot3_path)
    print(f"  [OK] Saved: {plot3_path.resolve()}")

    print("\n[SUCCESS] All Step 11 summary artifacts and plots successfully created!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate summary CSV and diagnostic plots for Step 11.")
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=Path("results/leaderboard.csv"),
        help="Path to raw leaderboard.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/leaderboard_summary.csv"),
        help="Destination for leaderboard_summary.csv",
    )
    parser.add_argument(
        "--outages-dir",
        type=Path,
        default=Path("data/processed/outages"),
        help="Directory containing outage parquets and manifest.csv",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("data/processed/cpp_predictions"),
        help="Directory containing C++ strapdown predictions",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/plots"),
        help="Destination directory for output PNG plots",
    )
    parser.add_argument(
        "--instance-short",
        type=str,
        default="pair_Vfa01__10s__0",
        help="Representative short outage instance (default: pair_Vfa01__10s__0)",
    )
    parser.add_argument(
        "--instance-long",
        type=str,
        default="pair_Y1__180s__0",
        help="Representative long outage instance (default: pair_Y1__180s__0)",
    )

    args = parser.parse_args()

    make_all_plots(
        leaderboard_csv=args.leaderboard_csv,
        summary_csv=args.summary_csv,
        outages_dir=args.outages_dir,
        prediction_dir=args.prediction_dir,
        plots_dir=args.plots_dir,
        instance_short=args.instance_short,
        instance_long=args.instance_long,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
