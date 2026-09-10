"""CLI runner for Module A static-phase attitude initialization and gyro bias estimation.

Processes all paired runs, writes results/module_a_static_phase.csv, and prints
a comprehensive statistical characterization of the dataset.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

from dataeval.harness.static_phase import (
    DEFAULT_ACCEL_STD_THRESH_M_S2,
    DEFAULT_MAX_DURATION_S,
    DEFAULT_MIN_DURATION_S,
    DEFAULT_SPEED_THRESH_KMH,
    estimate_static_phase,
)


def run_static_phase_batch(
    paired_dir: Path,
    output_csv: Path,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    speed_thresh_kmh: float = DEFAULT_SPEED_THRESH_KMH,
    accel_std_thresh: float = DEFAULT_ACCEL_STD_THRESH_M_S2,
) -> pd.DataFrame:
    """Execute static phase estimation across all paired Parquet files."""
    parquet_files = sorted(paired_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {paired_dir}")

    results = []
    print(f"Analyzing {len(parquet_files)} paired runs for static attitude and gyro bias...")

    for p in parquet_files:
        run_id = p.stem
        df = pd.read_parquet(p)
        res = estimate_static_phase(
            df,
            run_id=run_id,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            speed_thresh_kmh=speed_thresh_kmh,
            accel_std_thresh=accel_std_thresh,
        )
        results.append({
            "run_id": res.run_id,
            "window_found": res.window_found,
            "start_s": round(res.start_s, 2) if not np.isnan(res.start_s) else np.nan,
            "end_s": round(res.end_s, 2) if not np.isnan(res.end_s) else np.nan,
            "duration_s": round(res.duration_s, 2),
            "pitch_rad": round(res.pitch_rad, 6) if not np.isnan(res.pitch_rad) else np.nan,
            "roll_rad": round(res.roll_rad, 6) if not np.isnan(res.roll_rad) else np.nan,
            "pitch_deg": round(res.pitch_deg, 3) if not np.isnan(res.pitch_deg) else np.nan,
            "roll_deg": round(res.roll_deg, 3) if not np.isnan(res.roll_deg) else np.nan,
            "gyro_bias_x_rad_s": round(res.gyro_bias_x_rad_s, 8) if not np.isnan(res.gyro_bias_x_rad_s) else np.nan,
            "gyro_bias_y_rad_s": round(res.gyro_bias_y_rad_s, 8) if not np.isnan(res.gyro_bias_y_rad_s) else np.nan,
            "gyro_bias_z_rad_s": round(res.gyro_bias_z_rad_s, 8) if not np.isnan(res.gyro_bias_z_rad_s) else np.nan,
            "gyro_bias_x_deg_s": round(res.gyro_bias_x_deg_s, 4) if not np.isnan(res.gyro_bias_x_deg_s) else np.nan,
            "gyro_bias_y_deg_s": round(res.gyro_bias_y_deg_s, 4) if not np.isnan(res.gyro_bias_y_deg_s) else np.nan,
            "gyro_bias_z_deg_s": round(res.gyro_bias_z_deg_s, 4) if not np.isnan(res.gyro_bias_z_deg_s) else np.nan,
            "accel_mean_x_m_s2": round(res.accel_mean_x_m_s2, 4) if not np.isnan(res.accel_mean_x_m_s2) else np.nan,
            "accel_mean_y_m_s2": round(res.accel_mean_y_m_s2, 4) if not np.isnan(res.accel_mean_y_m_s2) else np.nan,
            "accel_mean_z_m_s2": round(res.accel_mean_z_m_s2, 4) if not np.isnan(res.accel_mean_z_m_s2) else np.nan,
            "accel_norm_m_s2": round(res.accel_norm_m_s2, 4) if not np.isnan(res.accel_norm_m_s2) else np.nan,
            "flag": res.flag,
        })

    out_df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header_docs = (
        "# Module A v1: Static-Phase Attitude and Gyroscope Bias Calibration\n"
        "# Coordinate Frame: ENU navigation, Body X=Right, Y=Forward, Z=Up\n"
        "# Leveling: pitch = atan2(fy, sqrt(fx^2 + fz^2)), roll = atan2(-fx, fz)\n"
        "# Units: pitch/roll in rad & deg; gyro biases in rad/s & deg/s; accel in m/s^2\n"
    )

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        f.write(header_docs)
        out_df.to_csv(f, index=False)

    print(f"\n[OK] Wrote static phase calibration results to: {output_csv}")
    return out_df


def print_summary(df: pd.DataFrame) -> None:
    """Print detailed characterization of static phase findings."""
    total = len(df)
    found = df["window_found"].sum()
    not_found = total - found

    print("\n" + "=" * 70)
    print("MODULE A STATIC-PHASE ATTITUDE & BIAS CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Total Paired Runs Analyzed : {total}")
    print(f"Stationary Window Found    : {found} ({found / total * 100:.1f}%)")
    print(f"No Window Found (Moving)   : {not_found} ({not_found / total * 100:.1f}%)")

    print("\nQuality Flag Breakdown:")
    for flag, cnt in df["flag"].value_counts().items():
        print(f"  - {flag:<16}: {cnt:2d} runs ({cnt / total * 100:.1f}%)")

    valid = df[df["window_found"]]

    print("\nStationary Window Durations (seconds):")
    durs = valid["duration_s"]
    print(f"  Min / 25% / Median / 75% / Max : {durs.min():.1f}s / {durs.quantile(0.25):.1f}s / {durs.median():.1f}s / {durs.quantile(0.75):.1f}s / {durs.max():.1f}s")
    print(f"  Total Static Time Logged       : {durs.sum():.1f}s ({durs.sum() / 60.0:.1f} min)")

    print("\nEstimated Initial Attitude Distribution (degrees):")
    p = valid["pitch_deg"]
    r = valid["roll_deg"]
    print(f"  Pitch (deg) : mean={p.mean():+.2f}°, std={p.std():.2f}°, min={p.min():+.2f}°, med={p.median():+.2f}°, max={p.max():+.2f}°")
    print(f"  Roll  (deg) : mean={r.mean():+.2f}°, std={r.std():.2f}°, min={r.min():+.2f}°, med={r.median():+.2f}°, max={r.max():+.2f}°")

    print("\nEstimated Gyroscope Biases (deg/s):")
    gx = valid["gyro_bias_x_deg_s"]
    gy = valid["gyro_bias_y_deg_s"]
    gz = valid["gyro_bias_z_deg_s"]
    print(f"  Gyro X (Lateral)      : mean={gx.mean():+.3f}°/s, std={gx.std():.3f}°/s, min={gx.min():+.3f}°/s, med={gx.median():+.3f}°/s, max={gx.max():+.3f}°/s")
    print(f"  Gyro Y (Longitudinal) : mean={gy.mean():+.3f}°/s, std={gy.std():.3f}°/s, min={gy.min():+.3f}°/s, med={gy.median():+.3f}°/s, max={gy.max():+.3f}°/s")
    print(f"  Gyro Z (Vertical/Yaw) : mean={gz.mean():+.3f}°/s, std={gz.std():.3f}°/s, min={gz.min():+.3f}°/s, med={gz.median():+.3f}°/s, max={gz.max():+.3f}°/s")

    # Flagged outliers
    p_outliers = valid[valid["pitch_deg"].abs() > 5.0]
    r_outliers = valid[valid["roll_deg"].abs() > 5.0]
    b_outliers = valid[(valid["gyro_bias_x_deg_s"].abs() > 1.0) | (valid["gyro_bias_y_deg_s"].abs() > 1.0) | (valid["gyro_bias_z_deg_s"].abs() > 1.0)]

    print("\nFlagged Outliers for Review:")
    print(f"  - |Pitch| > 5°: {len(p_outliers)} runs ({', '.join(p_outliers['run_id'].tolist()) if len(p_outliers) else 'None'})")
    print(f"  - |Roll|  > 5°: {len(r_outliers)} runs ({', '.join(r_outliers['run_id'].tolist()) if len(r_outliers) else 'None'})")
    print(f"  - |Bias|  > 1°/s: {len(b_outliers)} runs ({', '.join(b_outliers['run_id'].tolist()) if len(b_outliers) else 'None'})")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Module A static-phase attitude and bias calibration.")
    parser.add_argument("--paired-dir", type=Path, default=Path("data/processed/paired"), help="Path to paired parquet files.")
    parser.add_argument("--output-csv", type=Path, default=Path("results/module_a_static_phase.csv"), help="Path to output CSV.")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S, help="Min stationary duration in seconds.")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S, help="Max stationary duration in seconds.")
    parser.add_argument("--speed-thresh", type=float, default=DEFAULT_SPEED_THRESH_KMH, help="Speed threshold in km/h.")
    parser.add_argument("--accel-std-thresh", type=float, default=DEFAULT_ACCEL_STD_THRESH_M_S2, help="Accel std threshold in m/s^2.")
    args = parser.parse_args()

    df = run_static_phase_batch(
        paired_dir=args.paired_dir,
        output_csv=args.output_csv,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration,
        speed_thresh_kmh=args.speed_thresh,
        accel_std_thresh=args.accel_std_thresh,
    )
    print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
