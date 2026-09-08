"""Leaderboard summary aggregation engine.

Extracts ground-truth verified (paired-only) performance metrics from
results/leaderboard.csv and computes grouped summary statistics per (config, outage_s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "config",
    "outage_s",
    "n_paired",
    "median_final_pos_error_m",
    "p95_final_pos_error_m",
    "median_pct_of_distance",
    "p95_pct_of_distance",
    "median_heading_error_deg",
]


def compute_leaderboard_summary(
    leaderboard_path: Union[Path, str] = Path("results/leaderboard.csv"),
    output_path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """Compute summary statistics per (config, outage_s) from paired instances.

    Args:
        leaderboard_path: Path to raw results/leaderboard.csv.
        output_path: Optional destination to write results/leaderboard_summary.csv.

    Returns:
        DataFrame containing aggregated metrics for each (config, outage_s) tuple.

    Raises:
        FileNotFoundError: If leaderboard_path does not exist.
        ValueError: If leaderboard is empty or missing required columns.
    """
    path = Path(leaderboard_path)
    if not path.exists():
        raise FileNotFoundError(f"Leaderboard file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Leaderboard file is empty: {path}")

    required_raw_cols = [
        "config",
        "outage_s",
        "final_pos_error_m",
        "pct_of_distance",
        "heading_error_deg",
        "has_ground_truth",
    ]
    missing = [col for col in required_raw_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Leaderboard missing required columns: {missing}")

    # Strictly filter to paired instances with valid ground truth
    paired_df = df[df["has_ground_truth"] == True].copy()
    if paired_df.empty:
        raise ValueError("No paired instances with has_ground_truth==True found in leaderboard.")

    records = []
    # Group by config and outage_s preserving stable order
    grouped = paired_df.groupby(["config", "outage_s"], sort=False)

    for (config, outage_s), group in grouped:
        valid_pos = group["final_pos_error_m"].dropna()
        valid_pct = group["pct_of_distance"].dropna()
        valid_hdg = group["heading_error_deg"].dropna()

        n_paired = int(len(valid_pos))
        med_pos = float(valid_pos.median()) if not valid_pos.empty else np.nan
        p95_pos = float(valid_pos.quantile(0.95)) if not valid_pos.empty else np.nan
        med_pct = float(valid_pct.median()) if not valid_pct.empty else np.nan
        p95_pct = float(valid_pct.quantile(0.95)) if not valid_pct.empty else np.nan
        med_hdg = float(valid_hdg.median()) if not valid_hdg.empty else np.nan

        records.append({
            "config": str(config),
            "outage_s": int(outage_s) if float(outage_s).is_integer() else float(outage_s),
            "n_paired": n_paired,
            "median_final_pos_error_m": med_pos,
            "p95_final_pos_error_m": p95_pos,
            "median_pct_of_distance": med_pct,
            "p95_pct_of_distance": p95_pct,
            "median_heading_error_deg": med_hdg,
        })

    summary_df = pd.DataFrame(records)[SUMMARY_COLUMNS]
    summary_df = summary_df.sort_values(["config", "outage_s"]).reset_index(drop=True)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_p, index=False, float_format="%.4f")

    return summary_df


if __name__ == "__main__":
    out_csv = Path("results/leaderboard_summary.csv")
    summary = compute_leaderboard_summary(output_path=out_csv)
    print(f"Leaderboard summary written to {out_csv.resolve()}:")
    print(summary.to_string(index=False))
