"""CLI runner for Module B dynamic-phase yaw estimation and attitude synthesis.

Loads Step 14's static leveling and gyro bias calibrations, estimates PCA yaw
from early motion, performs gyro-Z cross-check, synthesizes initial attitude
quaternions for all 72 paired runs, and writes results/module_b_initial_attitude.csv.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from dataeval.harness.dynamic_phase import (
    DEFAULT_MIN_MOTION_DURATION_S,
    DEFAULT_MOTION_SPEED_THRESH_KMH,
    DEFAULT_WINDOW_LEN_S,
    estimate_dynamic_phase,
)


def load_static_calibrations(static_csv_path: Path) -> dict:
    """Load Step 14 calibration results keyed by run_id."""
    if not static_csv_path.exists():
        print(f"[WARN] Static calibrations file not found at {static_csv_path}. Proceeding with uncalibrated fallbacks.")
        return {}

    df = pd.read_csv(static_csv_path, comment="#")
    calibs = {}
    for _, row in df.iterrows():
        calibs[str(row["run_id"])] = {
            "window_found": bool(row.get("window_found", False)),
            "pitch_rad": float(row["pitch_rad"]) if pd.notna(row.get("pitch_rad")) else np.nan,
            "roll_rad": float(row["roll_rad"]) if pd.notna(row.get("roll_rad")) else np.nan,
            "gyro_bias_x_rad_s": float(row["gyro_bias_x_rad_s"]) if pd.notna(row.get("gyro_bias_x_rad_s")) else np.nan,
            "gyro_bias_y_rad_s": float(row["gyro_bias_y_rad_s"]) if pd.notna(row.get("gyro_bias_y_rad_s")) else np.nan,
            "gyro_bias_z_rad_s": float(row["gyro_bias_z_rad_s"]) if pd.notna(row.get("gyro_bias_z_rad_s")) else np.nan,
            "flag": str(row.get("flag", "")),
        }
    return calibs


def run_dynamic_phase_batch(
    paired_dir: Path,
    static_csv: Path,
    output_csv: Path,
    min_speed_kmh: float = DEFAULT_MOTION_SPEED_THRESH_KMH,
    min_duration_s: float = DEFAULT_MIN_MOTION_DURATION_S,
    window_len_s: float = DEFAULT_WINDOW_LEN_S,
) -> pd.DataFrame:
    """Execute dynamic yaw estimation and quaternion synthesis across all paired runs."""
    parquet_files = sorted(paired_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {paired_dir}")

    static_calibs = load_static_calibrations(static_csv)
    results = []
    print(f"Analyzing {len(parquet_files)} paired runs for dynamic yaw and full attitude quaternion...")

    for p in parquet_files:
        run_id = p.stem
        df = pd.read_parquet(p)
        static_rec = static_calibs.get(run_id)

        res = estimate_dynamic_phase(
            df,
            run_id=run_id,
            static_record=static_rec,
            min_speed_kmh=min_speed_kmh,
            min_duration_s=min_duration_s,
            window_len_s=window_len_s,
        )

        results.append({
            "run_id": res.run_id,
            "has_motion": res.has_motion,
            "yaw_rad": round(res.yaw_rad, 6) if not np.isnan(res.yaw_rad) else np.nan,
            "yaw_deg": round(res.yaw_deg, 3) if not np.isnan(res.yaw_deg) else np.nan,
            "yaw_source": res.yaw_source,
            "pitch_rad": round(res.pitch_rad, 6) if not np.isnan(res.pitch_rad) else np.nan,
            "roll_rad": round(res.roll_rad, 6) if not np.isnan(res.roll_rad) else np.nan,
            "pitch_deg": round(res.pitch_deg, 3) if not np.isnan(res.pitch_deg) else np.nan,
            "roll_deg": round(res.roll_deg, 3) if not np.isnan(res.roll_deg) else np.nan,
            "leveling_source": res.leveling_source,
            "gyro_bias_x_rad_s": round(res.gyro_bias_x_rad_s, 8) if not np.isnan(res.gyro_bias_x_rad_s) else np.nan,
            "gyro_bias_y_rad_s": round(res.gyro_bias_y_rad_s, 8) if not np.isnan(res.gyro_bias_y_rad_s) else np.nan,
            "gyro_bias_z_rad_s": round(res.gyro_bias_z_rad_s, 8) if not np.isnan(res.gyro_bias_z_rad_s) else np.nan,
            "gyro_bias_source": res.gyro_bias_source,
            "q_w": round(res.q_w, 7) if not np.isnan(res.q_w) else np.nan,
            "q_x": round(res.q_x, 7) if not np.isnan(res.q_x) else np.nan,
            "q_y": round(res.q_y, 7) if not np.isnan(res.q_y) else np.nan,
            "q_z": round(res.q_z, 7) if not np.isnan(res.q_z) else np.nan,
            "pca_heading_change_deg": round(res.pca_heading_change_deg, 3),
            "gyro_heading_change_deg": round(res.gyro_heading_change_deg, 3),
            "discrepancy_deg": round(res.discrepancy_deg, 3),
            "flag": res.flag,
        })

    out_df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header_docs = (
        "# Module B v1: Dynamic-Phase Yaw Estimation and Attitude Quaternion Synthesis\n"
        "# Navigation Frame: Local Tangent Plane East-North-Up (ENU)\n"
        "# Body Frame: X=Right, Y=Forward, Z=Up\n"
        "# Attitude Quaternion: q = [q_w, q_x, q_y, q_z] transforming v_nav = R(q) * v_body\n"
        "# Units: angles in rad & deg; gyro biases in rad/s; discrepancies in deg\n"
    )

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        f.write(header_docs)
        out_df.to_csv(f, index=False)

    print(f"\n[OK] Wrote initial attitude results to: {output_csv}")
    return out_df


def print_summary(df: pd.DataFrame) -> None:
    """Print complete statistical summary of dynamic yaw and synthesized attitude."""
    total = len(df)
    has_motion = df["has_motion"].sum()
    no_motion = total - has_motion

    print("\n" + "=" * 70)
    print("MODULE B DYNAMIC-PHASE YAW & INITIAL ATTITUDE SYNTHESIS SUMMARY")
    print("=" * 70)
    print(f"Total Paired Runs Analyzed : {total}")
    print(f"Dynamic Motion Found (Yaw) : {has_motion} ({has_motion / total * 100:.1f}%)")
    print(f"No Motion Found (Parked)   : {no_motion} ({no_motion / total * 100:.1f}%)")

    print("\nQuality Flag Breakdown:")
    for flag, cnt in df["flag"].value_counts().items():
        print(f"  - {flag:<24}: {cnt:2d} runs ({cnt / total * 100:.1f}%)")

    print("\nLeveling & Bias Source Breakdown:")
    for src, cnt in df["leveling_source"].value_counts().items():
        print(f"  - Leveling Source: {src:<20}: {cnt:2d} runs ({cnt / total * 100:.1f}%)")
    for src, cnt in df["gyro_bias_source"].value_counts().items():
        print(f"  - Gyro Bias Source: {src:<20}: {cnt:2d} runs ({cnt / total * 100:.1f}%)")

    moving = df[df["has_motion"]]

    print("\nEstimated Initial Yaw Distribution (degrees clockwise from North):")
    yaw = moving["yaw_deg"]
    print(f"  Yaw (deg) : mean={yaw.mean():.1f}°, std={yaw.std():.1f}°, min={yaw.min():.1f}°, med={yaw.median():.1f}°, max={yaw.max():.1f}°")

    print("\nGyro-Z Cross-Check Discrepancy (PCA delta - Gyro delta, degrees):")
    disc = moving["discrepancy_deg"]
    print(f"  Discrepancy (deg) : mean={disc.mean():+.2f}°, std={disc.std():.2f}°, min={disc.min():+.2f}°, med={disc.median():+.2f}°, max={disc.max():+.2f}°")
    print(f"  Discrepancy within ±5.0° : {(disc.abs() <= 5.0).sum()} / {len(moving)} ({(disc.abs() <= 5.0).mean() * 100:.1f}%)")
    print(f"  Discrepancy within ±10.0°: {(disc.abs() <= 10.0).sum()} / {len(moving)} ({(disc.abs() <= 10.0).mean() * 100:.1f}%)")

    # Step 14 high-bias runs inspection
    flagged = ["pair_S2", "pair_S3b", "pair_Vfa01", "pair_Vta17", "pair_Vta25"]
    print("\nDeep Inspection: 5 High-Gyro-Bias Runs Flagged in Step 14:")
    print(f"{'Run ID':<12} | {'PCA d_psi':<10} | {'Gyro d_psi':<10} | {'Discrepancy':<12} | {'Step 14 Bias Y':<15} | Status")
    print("-" * 75)
    flagged_df = moving[moving["run_id"].isin(flagged)]
    for _, r in flagged_df.iterrows():
        rid = r["run_id"]
        p_d = r["pca_heading_change_deg"]
        g_d = r["gyro_heading_change_deg"]
        d = r["discrepancy_deg"]
        note = "ELEVATED (Residual Motion)" if abs(d) > 10.0 else "Normal"
        print(f"{rid:<12} | {p_d:+8.2f}°  | {g_d:+8.2f}°  | {d:+10.2f}°  | {'flagged > 1°/s':<15} | {note}")

    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Module B dynamic-phase yaw estimation and attitude synthesis.")
    parser.add_argument("--paired-dir", type=Path, default=Path("data/processed/paired"), help="Path to paired parquet files.")
    parser.add_argument("--static-csv", type=Path, default=Path("results/module_a_static_phase.csv"), help="Path to Step 14 CSV.")
    parser.add_argument("--output-csv", type=Path, default=Path("results/module_b_initial_attitude.csv"), help="Path to output CSV.")
    parser.add_argument("--min-speed", type=float, default=DEFAULT_MOTION_SPEED_THRESH_KMH, help="Speed threshold in km/h.")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_MOTION_DURATION_S, help="Min duration in seconds.")
    parser.add_argument("--window-len", type=float, default=DEFAULT_WINDOW_LEN_S, help="Early motion window length in seconds.")
    args = parser.parse_args()

    df = run_dynamic_phase_batch(
        paired_dir=args.paired_dir,
        static_csv=args.static_csv,
        output_csv=args.output_csv,
        min_speed_kmh=args.min_speed,
        min_duration_s=args.min_duration,
        window_len_s=args.window_len,
    )
    print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
