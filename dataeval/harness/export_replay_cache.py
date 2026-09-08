"""Pre-export replay cache for C++ strapdown INS replay driver.

Reads outage-masked Parquet files from data/processed/outages/ and exports
minimal CSV files into data/processed/_cpp_replay_cache/ containing:
1. Header metadata: initial state (t0, lat0, lon0, alt0, speed_ms, heading_deg, ax0, ay0, az0)
   extracted using the exact Step 8 pre-outage circular averaging logic.
2. Canonical IMU sample rows across the outage duration:
   timestamp_s,ax,ay,az,gx,gy,gz,mx,my,mz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from dataeval.harness.baseline import (
    DEFAULT_PRE_OUTAGE_WINDOW_S,
    extract_pre_outage_state,
)
from dataeval.harness.outage import OutageWindow


def get_imu_columns(df: pd.DataFrame, source_side: str) -> Tuple[list[str], list[str]]:
    """Determine available IMU and magnetometer column names based on dataset schema."""
    if "phone_accel_x_m_s2" in df.columns:
        accel_cols = ["phone_accel_x_m_s2", "phone_accel_y_m_s2", "phone_accel_z_m_s2"]
        gyro_cols = ["phone_gyro_x_rad_s", "phone_gyro_y_rad_s", "phone_gyro_z_rad_s"]
        mag_cols = ["phone_mag_x_uT", "phone_mag_y_uT", "phone_mag_z_uT"]
    elif "accel_x_m_s2" in df.columns:
        accel_cols = ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]
        gyro_cols = ["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]
        mag_cols = ["mag_x_uT", "mag_y_uT", "mag_z_uT"]
    else:
        raise ValueError("Could not find standard accelerometer columns in DataFrame.")

    return accel_cols, gyro_cols, mag_cols


def get_pre_outage_altitude(df: pd.DataFrame, t0: float) -> float:
    """Extract altitude at or immediately prior to t0."""
    alt_col = None
    if "phone_altitude_m" in df.columns:
        alt_col = "phone_altitude_m"
    elif "altitude_m" in df.columns:
        alt_col = "altitude_m"
    elif "can_altitude_m" in df.columns:
        alt_col = "can_altitude_m"

    if alt_col and alt_col in df.columns:
        valid_alts = df.loc[df["timestamp_s"] <= t0, alt_col].dropna()
        if not valid_alts.empty:
            return float(valid_alts.iloc[-1])
    return 0.0


def export_single_instance(
    parquet_path: Path,
    output_path: Path,
    window: OutageWindow,
    pre_outage_window_s: float = DEFAULT_PRE_OUTAGE_WINDOW_S,
) -> bool:
    """Export a single outage Parquet file to the C++ replay CSV format."""
    if not parquet_path.exists():
        return False

    df = pd.read_parquet(parquet_path)
    if df.empty or "timestamp_s" not in df.columns:
        return False

    pre_state = extract_pre_outage_state(
        df=df,
        window=window,
        pre_outage_window_s=pre_outage_window_s,
    )
    if pre_state is None:
        return False

    alt0 = get_pre_outage_altitude(df, pre_state.t0)
    accel_cols, gyro_cols, mag_cols = get_imu_columns(df, window.source_side)

    # Initial acceleration: default to standard gravity reaction
    ax0, ay0, az0 = 0.0, 0.0, 9.80665

    # Outage slice: [start_s, end_s)
    outage_mask = (df["timestamp_s"] >= window.start_s) & (df["timestamp_s"] < window.end_s)
    outage_df = df[outage_mask].copy()
    if outage_df.empty:
        return False

    # Extract required IMU columns
    outage_df = outage_df.sort_values("timestamp_s")
    ts = outage_df["timestamp_s"].values
    ax = outage_df[accel_cols[0]].values
    ay = outage_df[accel_cols[1]].values
    az = outage_df[accel_cols[2]].values
    gx = outage_df[gyro_cols[0]].values
    gy = outage_df[gyro_cols[1]].values
    gz = outage_df[gyro_cols[2]].values

    # Optional magnetometer
    mx = outage_df[mag_cols[0]].values if mag_cols[0] in outage_df.columns else np.full(len(ts), np.nan)
    my = outage_df[mag_cols[1]].values if mag_cols[1] in outage_df.columns else np.full(len(ts), np.nan)
    mz = outage_df[mag_cols[2]].values if mag_cols[2] in outage_df.columns else np.full(len(ts), np.nan)

    # Write formatted CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Header comments containing initial condition parameters
        f.write(
            f"# initial_state: t0={pre_state.t0:.6f},lat0={pre_state.lat0:.8f},"
            f"lon0={pre_state.lon0:.8f},alt0={alt0:.3f},speed_ms={pre_state.v0:.4f},"
            f"heading_deg={pre_state.theta0:.4f},ax0={ax0:.4f},ay0={ay0:.4f},az0={az0:.4f}\n"
        )
        f.write("timestamp_s,ax,ay,az,gx,gy,gz,mx,my,mz\n")
        for i in range(len(ts)):
            f.write(
                f"{ts[i]:.6f},{ax[i]:.6f},{ay[i]:.6f},{az[i]:.6f},"
                f"{gx[i]:.6f},{gy[i]:.6f},{gz[i]:.6f},"
                f"{mx[i]:.6f},{my[i]:.6f},{mz[i]:.6f}\n"
            )

    return True


def export_all_instances(
    outages_dir: Path = Path("data/processed/outages"),
    output_dir: Path = Path("data/processed/_cpp_replay_cache"),
    pre_outage_window_s: float = DEFAULT_PRE_OUTAGE_WINDOW_S,
) -> int:
    """Export all valid outage instances from manifest to the C++ replay cache."""
    start_time = time.time()
    manifest_path = outages_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(manifest_path)
    ok_instances = manifest[manifest["status"] == "ok"].copy()
    if ok_instances.empty:
        print("No valid outage instances (status == 'ok') found.", file=sys.stderr)
        return 1

    print(f"Exporting C++ Replay Cache for {len(ok_instances)} outage instances:")
    print(f"  Outages directory: {outages_dir.resolve()}")
    print(f"  Cache directory:   {output_dir.resolve()}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for idx, row in ok_instances.iterrows():
        outage_id = str(row["outage_id"]).strip()
        run_or_pair_id = str(row["run_or_pair_id"]).strip()
        split = str(row["split"]).strip()
        source_side = str(row["source_side"]).strip()
        outage_s = float(row["outage_s"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])

        parquet_path = outages_dir / split / f"{outage_id}.parquet"
        cache_csv = output_dir / f"{outage_id}.csv"

        window = OutageWindow(
            run_id=run_or_pair_id,
            source_side=source_side,
            start_s=start_s,
            duration_s=outage_s,
            end_s=end_s,
        )

        ok = export_single_instance(
            parquet_path=parquet_path,
            output_path=cache_csv,
            window=window,
            pre_outage_window_s=pre_outage_window_s,
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1
            print(f"  [WARN] Failed to export {outage_id}")

    elapsed = time.time() - start_time
    print(f"\n[COMPLETED] Exported {success_count} instances ({fail_count} failed) in {elapsed:.2f}s")
    return 0 if fail_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Export C++ Replay Cache for GNSS Outages.")
    parser.add_argument(
        "--outages-dir",
        type=Path,
        default=Path("data/processed/outages"),
        help="Path to processed outages directory containing manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/_cpp_replay_cache"),
        help="Destination directory for replay cache CSV files.",
    )
    args = parser.parse_args()
    return export_all_instances(args.outages_dir, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
