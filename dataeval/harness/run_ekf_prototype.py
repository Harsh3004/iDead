"""Validation runner for Step 18: Python Prototype 15-State Error EKF with ZUPT + NHC.

Evaluates the filter against the named diagnostic instances:
- Problem cases that broke static correction: pair_Vw1 (mount shift), pair_Vta2 (bias doubled)
- Severe bias cases: pair_S2, pair_Vta17
- Already-good control cases: pair_Vfa02, pair_S3c
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
from dataeval.harness.run_corrected_replay import load_module_b_tiers


EARTH_RADIUS = 6371000.0


def compute_haversine_error_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in meters between two geodetic coordinates."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return float(EARTH_RADIUS * c)


def parse_cache_header(header_line: str) -> Dict[str, float]:
    """Parse # initial_state comment in replay cache CSV."""
    params: Dict[str, float] = {}
    prefix = "# initial_state:"
    if not header_line.startswith(prefix):
        return params
    for token in header_line[len(prefix):].split(","):
        if "=" in token:
            k, v = token.strip().split("=")
            params[k] = float(v)
    return params


def run_ekf_on_instance(
    outage_id: str,
    manifest_csv: Path = Path("data/processed/outages/manifest.csv"),
    cache_dir: Path = Path("data/processed/_cpp_replay_cache"),
    outages_dir: Path = Path("data/processed/outages"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
    ekf_config: Optional[EkfConfig] = None,
) -> Dict[str, Any]:
    """Run 15-state ES-EKF on a single outage instance and return detailed state traces."""
    manifest = pd.read_csv(manifest_csv).set_index("outage_id")
    if outage_id not in manifest.index:
        raise ValueError(f"Outage {outage_id} not found in manifest")

    m_row = manifest.loc[outage_id]
    run_id = str(m_row["run_or_pair_id"]).strip()
    split = str(m_row["split"]).strip()
    outage_s = int(m_row["outage_s"])
    source_side = str(m_row["source_side"]).strip()

    cache_file = cache_dir / f"{outage_id}.csv"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_file}")

    # Read initial state metadata
    with open(cache_file, "r") as fp:
        line1 = fp.readline().strip()
    init_params = parse_cache_header(line1)

    lat0 = init_params.get("lat0", 0.0)
    lon0 = init_params.get("lon0", 0.0)
    alt0 = init_params.get("alt0", 0.0)
    speed0 = init_params.get("speed_ms", 0.0)
    heading0 = init_params.get("heading_deg", 0.0)
    t0 = init_params.get("t0", 0.0)
    ax0 = init_params.get("ax0", 0.0)
    ay0 = init_params.get("ay0", 0.0)
    az0 = init_params.get("az0", 9.80665)
    init_accel = np.array([ax0, ay0, az0], dtype=float)

    # Initial attitude & bias from Module B
    df_att = pd.read_csv(attitude_csv, comment="#").set_index("run_id")
    q0 = None
    b_gyro0 = np.zeros(3, dtype=float)
    if run_id in df_att.index:
        row_att = df_att.loc[run_id]
        if pd.notna(row_att.get("q_w")):
            q0 = np.array([row_att["q_w"], row_att["q_x"], row_att["q_y"], row_att["q_z"]], dtype=float)
        if row_att.get("gyro_bias_source") == "module_a_measured":
            b_gyro0 = np.array([
                row_att["gyro_bias_x_rad_s"],
                row_att["gyro_bias_y_rad_s"],
                row_att["gyro_bias_z_rad_s"]
            ], dtype=float)

    # Initialize EKF
    ekf = ErrorStateEkf(config=ekf_config)
    ekf.initialize(
        t0=t0,
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        speed_ms=speed0,
        heading_deg=heading0,
        q0=q0,
        b_gyro0=b_gyro0,
        initial_accel=init_accel,
    )

    # Load IMU samples from cache
    imu_df = pd.read_csv(cache_file, comment="#")

    # Trace arrays
    trace_t: List[float] = []
    trace_lat: List[float] = []
    trace_lon: List[float] = []
    trace_speed: List[float] = []
    trace_heading: List[float] = []
    trace_cov_trace: List[float] = []
    trace_pos_uncertainty: List[float] = []
    trace_bg_x: List[float] = []
    trace_bg_y: List[float] = []
    trace_bg_z: List[float] = []
    trace_ba_x: List[float] = []
    trace_ba_y: List[float] = []
    trace_ba_z: List[float] = []
    trace_zupt_applied: List[bool] = []
    trace_nhc_applied: List[bool] = []

    for _, row in imu_df.iterrows():
        t = float(row["timestamp_s"])
        f_b = np.array([float(row["ax"]), float(row["ay"]), float(row["az"])], dtype=float)
        omega_b = np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])], dtype=float)

        # 1. Prediction step
        ekf.predict(t, f_b, omega_b)

        # 2. Measurement updates
        applied_zupt = False
        applied_nhc = False

        if ekf.is_stationary():
            applied_zupt = ekf.update_zupt()
        else:
            # Apply NHC if moving with forward speed > 0.5 m/s
            if ekf.get_speed_ms() > 0.5:
                applied_nhc = ekf.update_nhc()

        lat, lon, alt = ekf.get_position_lat_lon()
        trace_t.append(t)
        trace_lat.append(lat)
        trace_lon.append(lon)
        trace_speed.append(ekf.get_speed_ms())
        trace_heading.append(ekf.get_heading_deg())
        trace_cov_trace.append(ekf.get_covariance_trace())
        trace_pos_uncertainty.append(ekf.get_position_uncertainty_m())
        trace_bg_x.append(float(ekf.b_gyro[0]))
        trace_bg_y.append(float(ekf.b_gyro[1]))
        trace_bg_z.append(float(ekf.b_gyro[2]))
        trace_ba_x.append(float(ekf.b_accel[0]))
        trace_ba_y.append(float(ekf.b_accel[1]))
        trace_ba_z.append(float(ekf.b_accel[2]))
        trace_zupt_applied.append(applied_zupt)
        trace_nhc_applied.append(applied_nhc)

    # Load Ground Truth
    outage_file = outages_dir / split / f"{outage_id}.parquet"
    parquet_df = pd.read_parquet(outage_file)
    window = OutageWindow(
        run_id=run_id,
        source_side=source_side,
        start_s=float(m_row["start_s"]),
        duration_s=float(outage_s),
        end_s=float(m_row["end_s"]),
    )
    gt_df = ground_truth_trajectory(parquet_df, window)
    if gt_df is None or len(gt_df) < 2:
        raise RuntimeError(f"Ground truth missing for {outage_id}")

    final_gt_lat = float(gt_df["latitude_deg"].iloc[-1])
    final_gt_lon = float(gt_df["longitude_deg"].iloc[-1])

    final_ekf_lat = trace_lat[-1]
    final_ekf_lon = trace_lon[-1]
    final_ekf_error_m = compute_haversine_error_m(final_ekf_lat, final_ekf_lon, final_gt_lat, final_gt_lon)

    # Load Bare & Corrected baseline predictions for comparison
    bare_file = Path("data/processed/cpp_predictions") / f"{outage_id}.csv"
    corr_file = Path("data/processed/cpp_predictions_corrected") / f"{outage_id}.csv"

    bare_error_m = np.nan
    corr_error_m = np.nan
    if bare_file.exists():
        bare_df = load_cpp_prediction(bare_file)
        bare_error_m = compute_haversine_error_m(
            float(bare_df["latitude_deg"].iloc[-1]),
            float(bare_df["longitude_deg"].iloc[-1]),
            final_gt_lat,
            final_gt_lon
        )
    if corr_file.exists():
        corr_df = load_cpp_prediction(corr_file)
        corr_error_m = compute_haversine_error_m(
            float(corr_df["latitude_deg"].iloc[-1]),
            float(corr_df["longitude_deg"].iloc[-1]),
            final_gt_lat,
            final_gt_lon
        )

    trace_df = pd.DataFrame({
        "timestamp_s": trace_t,
        "latitude_deg": trace_lat,
        "longitude_deg": trace_lon,
        "speed_ms": trace_speed,
        "heading_deg": trace_heading,
        "cov_trace": trace_cov_trace,
        "pos_uncertainty_m": trace_pos_uncertainty,
        "bg_x_rad_s": trace_bg_x,
        "bg_y_rad_s": trace_bg_y,
        "bg_z_rad_s": trace_bg_z,
        "ba_x_m_s2": trace_ba_x,
        "ba_y_m_s2": trace_ba_y,
        "ba_z_m_s2": trace_ba_z,
        "zupt_applied": trace_zupt_applied,
        "nhc_applied": trace_nhc_applied,
    })

    return {
        "outage_id": outage_id,
        "run_id": run_id,
        "outage_s": outage_s,
        "bare_error_m": bare_error_m,
        "corr_error_m": corr_error_m,
        "ekf_error_m": final_ekf_error_m,
        "zupt_count": sum(trace_zupt_applied),
        "nhc_count": sum(trace_nhc_applied),
        "final_cov_trace": trace_cov_trace[-1],
        "trace_df": trace_df,
    }


def main():
    print("=" * 80)
    print("STEP 18: PYTHON PROTOTYPE 15-STATE ERROR EKF VALIDATION")
    print("=" * 80)

    # Named target instances specified in task brief:
    target_instances = [
        # Problem cases that broke static correction
        "pair_Vw1__180s__0",    # mount shift / physical re-settling mid-trip (+920% in Step 16)
        "pair_Vta2__180s__0",   # gyro bias doubled (+661% in Step 16)
        # Severe physical bias cases
        "pair_S2__180s__0",     # severe bias (-6 km in Step 16, huge bias in Module A)
        "pair_S2__120s__0",     # severe bias (-33 km in Step 16)
        "pair_Vta17__180s__0",  # severe bias (-40 km in Step 16)
        "pair_Vta17__60s__0",   # severe bias (-6 km in Step 16)
        # Control cases that already performed well (must not regress)
        "pair_Vfa02__180s__0",  # -71.7% in Step 16
        "pair_S3c__180s__0",    # -85.3% in Step 16
    ]

    results = []
    detailed_traces = {}

    for oid in target_instances:
        print(f"\nEvaluating: {oid}...")
        res = run_ekf_on_instance(oid)
        results.append({
            "outage_id": res["outage_id"],
            "outage_s": res["outage_s"],
            "bare_error_m": res["bare_error_m"],
            "corr_error_m": res["corr_error_m"],
            "ekf_error_m": res["ekf_error_m"],
            "delta_vs_corr_m": res["ekf_error_m"] - res["corr_error_m"],
            "pct_vs_corr": ((res["ekf_error_m"] - res["corr_error_m"]) / res["corr_error_m"]) * 100.0,
            "zupt_updates": res["zupt_count"],
            "nhc_updates": res["nhc_count"],
            "cov_trace": res["final_cov_trace"],
        })
        detailed_traces[oid] = res["trace_df"]

    summary_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION SUMMARY TABLE")
    print("=" * 80)
    cols_show = ["outage_id", "outage_s", "bare_error_m", "corr_error_m", "ekf_error_m", "delta_vs_corr_m", "pct_vs_corr", "zupt_updates", "nhc_updates"]
    print(summary_df[cols_show].to_string(index=False))

    # Detailed state traces for pair_Vw1 and pair_Vta2
    for oid in ["pair_Vw1__180s__0", "pair_Vta2__180s__0"]:
        if oid in detailed_traces:
            tdf = detailed_traces[oid]
            print(f"\nState Trace Summary for {oid} (10 samples sampled across outage):")
            sample_indices = np.linspace(0, len(tdf) - 1, 10, dtype=int)
            sample_df = tdf.iloc[sample_indices][[
                "timestamp_s", "speed_ms", "heading_deg", "cov_trace",
                "bg_x_rad_s", "bg_y_rad_s", "bg_z_rad_s",
                "ba_x_m_s2", "ba_y_m_s2", "ba_z_m_s2"
            ]]
            print(sample_df.to_string(index=False))


if __name__ == "__main__":
    main()
