"""Diagnostic analysis for Step 17: Diagnosing the calibrated-tier reversal at 60s/180s.

Investigates:
1. Full trajectory error vs. time curves (crossover timing).
2. Gyro bias stability across multi-stillness intervals.
3. In-outage pitch/roll variations and horizontal gravity leakage.
4. Per-instance culprit ranking and calibration-outage time separation.
5. Ablation on suspect runs pair_S2 and pair_Vta17 (yaw vs. bias effect).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
from dataeval.harness.run_corrected_replay import load_module_b_tiers


EARTH_RADIUS = 6371000.0


def haversine_distance_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in meters."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS * c


def compute_trajectory_error_curves(
    manifest_csv: Path = Path("data/processed/outages/manifest.csv"),
    outages_dir: Path = Path("data/processed/outages"),
    bare_dir: Path = Path("data/processed/cpp_predictions"),
    corr_dir: Path = Path("data/processed/cpp_predictions_corrected"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
) -> Dict[int, Dict[str, Any]]:
    """Compute trajectory error vs. normalized time curves across outage durations."""
    tiers = load_module_b_tiers(attitude_csv)
    calib_runs = {r for r, v in tiers.items() if v["tier"] == "module_a_measured"}

    manifest = pd.read_csv(manifest_csv)
    calib_manifest = manifest[
        (manifest["status"] == "ok")
        & (manifest["has_cross_stream_ground_truth"] == True)
        & (manifest["run_or_pair_id"].isin(calib_runs))
    ].copy()

    durations = [10, 30, 60, 120, 180]
    time_fractions = np.linspace(0.0, 1.0, 21) # 5% steps: 0, 0.05, ..., 1.0
    results: Dict[int, Dict[str, Any]] = {}

    for d in durations:
        sub = calib_manifest[calib_manifest["outage_s"] == d]
        bare_curves = []
        corr_curves = []

        for _, row in sub.iterrows():
            oid = str(row["outage_id"]).strip()
            split = str(row["split"]).strip()
            run_id = str(row["run_or_pair_id"]).strip()

            bare_file = bare_dir / f"{oid}.csv"
            corr_file = corr_dir / f"{oid}.csv"
            outage_file = outages_dir / split / f"{oid}.parquet"

            if not bare_file.exists() or not corr_file.exists() or not outage_file.exists():
                continue

            bare_df = load_cpp_prediction(bare_file)
            corr_df = load_cpp_prediction(corr_file)
            parquet_df = pd.read_parquet(outage_file)

            window = OutageWindow(
                run_id=run_id,
                source_side=str(row["source_side"]).strip(),
                start_s=float(row["start_s"]),
                duration_s=float(d),
                end_s=float(row["end_s"]),
            )
            gt_df = ground_truth_trajectory(parquet_df, window)
            if gt_df is None or len(gt_df) < 2:
                continue

            # Align timestamps
            n_ticks = min(len(bare_df), len(corr_df), len(gt_df))
            if n_ticks < 5:
                continue

            lat_gt = gt_df["latitude_deg"].values[:n_ticks]
            lon_gt = gt_df["longitude_deg"].values[:n_ticks]

            lat_bare = bare_df["latitude_deg"].values[:n_ticks]
            lon_bare = bare_df["longitude_deg"].values[:n_ticks]

            lat_corr = corr_df["latitude_deg"].values[:n_ticks]
            lon_corr = corr_df["longitude_deg"].values[:n_ticks]

            err_bare = haversine_distance_m(lat_bare, lon_bare, lat_gt, lon_gt)
            err_corr = haversine_distance_m(lat_corr, lon_corr, lat_gt, lon_gt)

            # Interpolate to uniform time fractions
            t_orig = np.linspace(0.0, 1.0, n_ticks)
            bare_interp = np.interp(time_fractions, t_orig, err_bare)
            corr_interp = np.interp(time_fractions, t_orig, err_corr)

            bare_curves.append(bare_interp)
            corr_curves.append(corr_interp)

        bare_mat = np.array(bare_curves)
        corr_mat = np.array(corr_curves)

        med_bare = np.median(bare_mat, axis=0) if len(bare_curves) > 0 else np.zeros_like(time_fractions)
        med_corr = np.median(corr_mat, axis=0) if len(corr_curves) > 0 else np.zeros_like(time_fractions)

        # Find crossover fraction where corr > bare
        crossover_idx = np.where(med_corr > med_bare)[0]
        crossover_frac = time_fractions[crossover_idx[0]] if len(crossover_idx) > 0 else 1.0
        crossover_sec = crossover_frac * d

        results[d] = {
            "n": len(bare_curves),
            "time_fractions": time_fractions,
            "median_bare": med_bare,
            "median_corr": med_corr,
            "crossover_frac": crossover_frac,
            "crossover_sec": crossover_sec,
        }

    return results


def check_gyro_bias_stability(
    paired_dir: Path = Path("data/processed/paired"),
    static_phase_csv: Path = Path("results/module_a_static_phase.csv"),
) -> Dict[str, Any]:
    """Examine stillness windows and detect subsequent stationary intervals to evaluate drift."""
    df_static = pd.read_csv(static_phase_csv, comment="#")
    calib_static = df_static[df_static["window_found"] == True].copy()

    drift_records = []
    runs_with_multi_stops = 0

    for _, row in calib_static.iterrows():
        run_id = str(row["run_id"])
        parquet_file = paired_dir / f"{run_id}.parquet"
        if not parquet_file.exists():
            continue

        p_df = pd.read_parquet(parquet_file)
        # Find speed column
        speed_col = None
        for cand in ["can_vehicle_speed_kmh", "can_wheel_speed_kmh", "phone_gps_speed_kmh"]:
            if cand in p_df.columns and not p_df[cand].isna().all():
                speed_col = cand
                break
        if not speed_col:
            continue

        t_start_1 = float(row["start_s"])
        t_end_1 = float(row["end_s"])
        dur_1 = float(row["duration_s"])

        # Detect any subsequent still window of >= 3 seconds occurring at least 30s after window 1
        speed_series = p_df[speed_col].fillna(999.0)
        time_series = p_df["timestamp_s"]

        is_still = (speed_series < 0.5).values
        change_indices = np.where(np.diff(is_still.astype(int)) != 0)[0] + 1
        splits = np.split(np.arange(len(is_still)), change_indices)

        subsequent_windows = []
        for s in splits:
            if len(s) == 0 or not is_still[s[0]]:
                continue
            t_s = float(time_series.iloc[s[0]])
            t_e = float(time_series.iloc[s[-1]])
            dur = t_e - t_s
            # Ensure window is >= 3.0s and separated from window 1
            if dur >= 3.0 and (t_s > t_end_1 + 20.0 or t_e < t_start_1 - 20.0):
                subsequent_windows.append((t_s, t_e, dur))

        if len(subsequent_windows) > 0:
            runs_with_multi_stops += 1
            # Pick the longest subsequent window
            subsequent_windows.sort(key=lambda x: x[2], reverse=True)
            t_s2, t_e2, dur_2 = subsequent_windows[0]

            sub_w1 = p_df[(time_series >= t_start_1) & (time_series <= t_end_1)]
            sub_w2 = p_df[(time_series >= t_s2) & (time_series <= t_e2)]

            g1_x = sub_w1["phone_gyro_x_rad_s"].mean()
            g1_y = sub_w1["phone_gyro_y_rad_s"].mean()
            g1_z = sub_w1["phone_gyro_z_rad_s"].mean()

            g2_x = sub_w2["phone_gyro_x_rad_s"].mean()
            g2_y = sub_w2["phone_gyro_y_rad_s"].mean()
            g2_z = sub_w2["phone_gyro_z_rad_s"].mean()

            a1_x = sub_w1["phone_accel_x_m_s2"].mean()
            a1_y = sub_w1["phone_accel_y_m_s2"].mean()
            a1_z = sub_w1["phone_accel_z_m_s2"].mean()

            a2_x = sub_w2["phone_accel_x_m_s2"].mean()
            a2_y = sub_w2["phone_accel_y_m_s2"].mean()
            a2_z = sub_w2["phone_accel_z_m_s2"].mean()

            time_separation_s = abs(t_s2 - t_start_1)
            delta_gyro_x = g2_x - g1_x
            delta_gyro_y = g2_y - g1_y
            delta_gyro_z = g2_z - g1_z
            gyro_shift_norm = np.sqrt(delta_gyro_x**2 + delta_gyro_y**2 + delta_gyro_z**2)

            delta_accel_x = a2_x - a1_x
            delta_accel_y = a2_y - a1_y
            delta_accel_z = a2_z - a1_z
            accel_shift_norm = np.sqrt(delta_accel_x**2 + delta_accel_y**2 + delta_accel_z**2)

            drift_records.append({
                "run_id": run_id,
                "time_sep_s": time_separation_s,
                "dur1_s": dur_1,
                "dur2_s": dur_2,
                "g1_norm_rad_s": np.sqrt(g1_x**2 + g1_y**2 + g1_z**2),
                "g2_norm_rad_s": np.sqrt(g2_x**2 + g2_y**2 + g2_z**2),
                "gyro_shift_norm_rad_s": gyro_shift_norm,
                "gyro_shift_norm_deg_s": np.degrees(gyro_shift_norm),
                "accel_shift_norm_m_s2": accel_shift_norm,
                "pitch_shift_deg": np.degrees(np.arctan2(a2_y, a2_z) - np.arctan2(a1_y, a1_z)),
                "roll_shift_deg": np.degrees(np.arctan2(-a2_x, a2_z) - np.arctan2(-a1_x, a1_z)),
            })

    drift_df = pd.DataFrame(drift_records)
    return {
        "total_calibrated_runs": len(calib_static),
        "runs_with_multi_stops": runs_with_multi_stops,
        "drift_df": drift_df,
    }


def check_accelerometer_and_gravity_leakage(
    manifest_csv: Path = Path("data/processed/outages/manifest.csv"),
    paired_dir: Path = Path("data/processed/paired"),
    static_phase_csv: Path = Path("results/module_a_static_phase.csv"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
) -> pd.DataFrame:
    """Quantify road grade / dynamic tilt changes during 60s and 180s outages."""
    tiers = load_module_b_tiers(attitude_csv)
    calib_runs = {r for r, v in tiers.items() if v["tier"] == "module_a_measured"}

    df_static = pd.read_csv(static_phase_csv, comment="#").set_index("run_id")
    manifest = pd.read_csv(manifest_csv)

    leakage_records = []
    target_durations = [60, 180]

    for d in target_durations:
        sub = manifest[
            (manifest["outage_s"] == d)
            & (manifest["status"] == "ok")
            & (manifest["has_cross_stream_ground_truth"] == True)
            & (manifest["run_or_pair_id"].isin(calib_runs))
        ]

        for _, row in sub.iterrows():
            oid = str(row["outage_id"]).strip()
            run_id = str(row["run_or_pair_id"]).strip()
            t_start = float(row["start_s"])
            t_end = float(row["end_s"])

            parquet_file = paired_dir / f"{run_id}.parquet"
            if not parquet_file.exists():
                continue

            p_df = pd.read_parquet(parquet_file)
            sub_out = p_df[(p_df["timestamp_s"] >= t_start) & (p_df["timestamp_s"] <= t_end)]
            if len(sub_out) < 10:
                continue

            # Initial calibration values
            static_pitch_rad = float(df_static.loc[run_id, "pitch_rad"]) if run_id in df_static.index else 0.0
            static_roll_rad = float(df_static.loc[run_id, "roll_rad"]) if run_id in df_static.index else 0.0

            # Compute smoothed dynamic tilt during outage from low-pass filtered accel
            # phone_accel has high-frequency vibration; roll mean over 2.0s (20 samples) isolates tilt/grade
            ax_smooth = sub_out["phone_accel_x_m_s2"].rolling(20, min_periods=1).mean()
            ay_smooth = sub_out["phone_accel_y_m_s2"].rolling(20, min_periods=1).mean()
            az_smooth = sub_out["phone_accel_z_m_s2"].rolling(20, min_periods=1).mean()

            # Dynamic pitch: atan2(ay, sqrt(ax^2 + az^2)), dynamic roll: atan2(-ax, az)
            dyn_pitch = np.arctan2(ay_smooth, np.sqrt(ax_smooth**2 + az_smooth**2))
            dyn_roll = np.arctan2(-ax_smooth, az_smooth)

            pitch_dev = dyn_pitch - static_pitch_rad
            roll_dev = dyn_roll - static_roll_rad
            tilt_dev = np.sqrt(pitch_dev**2 + roll_dev**2)

            mean_tilt_dev_deg = float(np.degrees(tilt_dev.mean()))
            max_tilt_dev_deg = float(np.degrees(tilt_dev.max()))

            # Gravity leakage acceleration ~ g * sin(tilt_dev) ~ g * tilt_dev
            a_leak_mean = 9.80665 * np.sin(tilt_dev).mean()
            # Predicted position drift from gravity leakage alone over outage duration d: 0.5 * a * d^2
            theoretical_leakage_drift_m = 0.5 * a_leak_mean * (d ** 2)

            leakage_records.append({
                "outage_id": oid,
                "run_id": run_id,
                "outage_s": d,
                "mean_tilt_dev_deg": mean_tilt_dev_deg,
                "max_tilt_dev_deg": max_tilt_dev_deg,
                "a_leak_mean_m_s2": a_leak_mean,
                "theoretical_leakage_drift_m": theoretical_leakage_drift_m,
            })

    return pd.DataFrame(leakage_records)


def run_ablation_suspect_runs(
    manifest_csv: Path = Path("data/processed/outages/manifest.csv"),
    cache_dir: Path = Path("data/processed/_cpp_replay_cache"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
    outages_dir: Path = Path("data/processed/outages"),
) -> pd.DataFrame:
    """Perform exact 4-way ablation on suspect runs pair_S2 and pair_Vta17 to decouple yaw vs. bias effect."""
    from dataeval.harness.dynamic_phase import construct_attitude_quaternion
    from dataeval.harness.metrics import compute_outage_metrics

    # Python implementation of strapdown INS matching core C++ engine for exact ablation
    class PyStrapdown:
        def __init__(self):
            self.kGravity = 9.80665
            self.kEarthRadius = 6371000.0

        def init_state(self, lat0, lon0, speed_ms, heading_deg, q0, gyro_bias):
            self.lat0 = lat0
            self.lon0 = lon0
            self.q = np.array(q0, dtype=float) # w, x, y, z
            self.gyro_bias = np.array(gyro_bias, dtype=float)
            psi = np.radians(heading_deg)
            self.v = np.array([speed_ms * np.sin(psi), speed_ms * np.cos(psi), 0.0])
            self.p = np.array([0.0, 0.0, 0.0])
            self.last_accel_nav = np.array([0.0, 0.0, 0.0])

        def quat_rotate(self, q, v):
            w, x, y, z = q
            qv = np.array([x, y, z])
            t = 2.0 * np.cross(qv, v)
            return v + w * t + np.cross(qv, t)

        def quat_propagate(self, q, omega, dt):
            rot_vec = omega * dt
            theta = np.linalg.norm(rot_vec)
            if theta < 1e-12:
                dq = np.array([1.0, 0.5 * rot_vec[0], 0.5 * rot_vec[1], 0.5 * rot_vec[2]])
            else:
                half_theta = 0.5 * theta
                factor = np.sin(half_theta) / theta
                dq = np.array([np.cos(half_theta), rot_vec[0] * factor, rot_vec[1] * factor, rot_vec[2] * factor])
            # quat mult q * dq
            w1, x1, y1, z1 = q
            w2, x2, y2, z2 = dq
            q_new = np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])
            return q_new / np.linalg.norm(q_new)

        def run(self, imu_samples, dt=0.1):
            traj = []
            for s in imu_samples:
                # Gyro bias subtraction
                omega = np.array([s["gx"] - self.gyro_bias[0], s["gy"] - self.gyro_bias[1], s["gz"] - self.gyro_bias[2]])
                self.q = self.quat_propagate(self.q, omega, dt)
                # Specific force resolution
                f_body = np.array([s["ax"], s["ay"], s["az"]])
                f_nav = self.quat_rotate(self.q, f_body)
                accel_nav = f_nav + np.array([0.0, 0.0, -self.kGravity])
                # Trapezoidal integration
                delta_v = (self.last_accel_nav + accel_nav) * (0.5 * dt)
                v_prev = self.v.copy()
                self.v += delta_v
                self.p += (v_prev + self.v) * (0.5 * dt)
                self.last_accel_nav = accel_nav
                # Position
                delta_lat = (self.p[1] / self.kEarthRadius) * (180.0 / np.pi)
                delta_lon = (self.p[0] / (self.kEarthRadius * np.cos(np.radians(self.lat0)))) * (180.0 / np.pi)
                traj.append((self.lat0 + delta_lat, self.lon0 + delta_lon))
            return traj

    manifest = pd.read_csv(manifest_csv)
    df_att = pd.read_csv(attitude_csv, comment="#").set_index("run_id")

    ablation_records = []
    suspect_runs = ["pair_S2", "pair_Vta17"]

    for r in suspect_runs:
        row_att = df_att.loc[r]
        q_corr = (float(row_att["q_w"]), float(row_att["q_x"]), float(row_att["q_y"]), float(row_att["q_z"]))
        bias_corr = (float(row_att["gyro_bias_x_rad_s"]), float(row_att["gyro_bias_y_rad_s"]), float(row_att["gyro_bias_z_rad_s"]))

        sub_m = manifest[(manifest["run_or_pair_id"] == r) & (manifest["status"] == "ok")].sort_values("outage_s")

        for _, row in sub_m.iterrows():
            oid = str(row["outage_id"]).strip()
            outage_s = int(row["outage_s"])
            split = str(row["split"]).strip()

            cache_file = cache_dir / f"{oid}.csv"
            outage_file = outages_dir / split / f"{oid}.parquet"
            if not cache_file.exists() or not outage_file.exists():
                continue

            # Read initial state from cache file header
            with open(cache_file, "r") as fp:
                line1 = fp.readline().strip()
            # parse initial_state
            params = {}
            for token in line1.replace("# initial_state:", "").split(","):
                if "=" in token:
                    k, v = token.strip().split("=")
                    params[k] = float(v)

            lat0 = params["lat0"]
            lon0 = params["lon0"]
            speed0 = params["speed_ms"]
            heading0 = params["heading_deg"]

            # Load IMU samples from cache
            c_df = pd.read_csv(cache_file, comment="#")
            samples = []
            for _, c_row in c_df.iterrows():
                samples.append({
                    "ax": float(c_row["ax"]),
                    "ay": float(c_row["ay"]),
                    "az": float(c_row["az"]),
                    "gx": float(c_row["gx"]),
                    "gy": float(c_row["gy"]),
                    "gz": float(c_row["gz"]),
                })

            # Load ground truth
            p_df = pd.read_parquet(outage_file)
            window = OutageWindow(
                run_id=r,
                source_side=str(row["source_side"]).strip(),
                start_s=float(row["start_s"]),
                duration_s=float(outage_s),
                end_s=float(row["end_s"]),
            )
            gt_df = ground_truth_trajectory(p_df, window)
            if gt_df is None or len(gt_df) < 2:
                continue

            gt_lat_end = gt_df["latitude_deg"].iloc[-1]
            gt_lon_end = gt_df["longitude_deg"].iloc[-1]

            # 4 Configurations:
            # 1. Bare baseline: q from heading0 (flat), bias = [0,0,0]
            q_bare = construct_attitude_quaternion(yaw_rad=np.radians(heading0), pitch_rad=0.0, roll_rad=0.0)
            bias_bare = (0.0, 0.0, 0.0)

            # 2. Corrected Yaw Only: q from heading0 + pitch/roll=0, bias = [0,0,0]
            # (or q from Module B yaw + zero pitch/roll)
            q_yaw_only = construct_attitude_quaternion(yaw_rad=float(row_att["yaw_rad"]), pitch_rad=0.0, roll_rad=0.0)

            # 3. Corrected Bias Only: q from bare heading0, bias from Module A
            # bias_corr

            # 4. Full Corrected: q_corr, bias_corr

            configs = {
                "bare": (q_bare, bias_bare),
                "yaw_only": (q_yaw_only, bias_bare),
                "bias_only": (q_bare, bias_corr),
                "full_corrected": (q_corr, bias_corr),
            }

            errors = {}
            for c_name, (q_cfg, b_cfg) in configs.items():
                ins = PyStrapdown()
                ins.init_state(lat0, lon0, speed0, heading0, q_cfg, b_cfg)
                traj = ins.run(samples, dt=0.1)
                lat_end, lon_end = traj[-1]
                err = haversine_distance_m(np.array([lat_end]), np.array([lon_end]), np.array([gt_lat_end]), np.array([gt_lon_end]))[0]
                errors[c_name] = err

            ablation_records.append({
                "run_id": r,
                "outage_id": oid,
                "outage_s": outage_s,
                "err_bare_m": errors["bare"],
                "err_yaw_only_m": errors["yaw_only"],
                "err_bias_only_m": errors["bias_only"],
                "err_full_corrected_m": errors["full_corrected"],
            })

    return pd.DataFrame(ablation_records)


def analyze_outlier_runs(
    leaderboard_csv: Path = Path("results/leaderboard.csv"),
    static_phase_csv: Path = Path("results/module_a_static_phase.csv"),
    manifest_csv: Path = Path("data/processed/outages/manifest.csv"),
    attitude_csv: Path = Path("results/module_b_initial_attitude.csv"),
) -> pd.DataFrame:
    """Analyze the top degraded outlier runs at 60s and 180s."""
    df_lb = pd.read_csv(leaderboard_csv)
    tiers = load_module_b_tiers(attitude_csv)
    df_lb["tier"] = df_lb["run_id"].map(lambda r: tiers.get(r, {}).get("tier", "unknown"))
    calib = df_lb[df_lb["tier"] == "module_a_measured"]

    bare = calib[calib["config"] == "baseline_cpp_strapdown_v1"].set_index("outage_id")
    corr = calib[calib["config"] == "corrected_init_strapdown_v1"].set_index("outage_id")

    df_static = pd.read_csv(static_phase_csv, comment="#").set_index("run_id")
    manifest = pd.read_csv(manifest_csv).set_index("outage_id")

    records = []
    for d in [60, 180]:
        b_sub = bare[bare["outage_s"] == d]
        c_sub = corr[corr["outage_s"] == d]
        for oid in b_sub.index:
            if oid not in c_sub.index or oid not in manifest.index:
                continue
            err_b = float(b_sub.loc[oid, "final_pos_error_m"])
            err_c = float(c_sub.loc[oid, "final_pos_error_m"])
            diff = err_c - err_b
            run_id = str(b_sub.loc[oid, "run_id"])
            m_row = manifest.loc[oid]
            outage_start = float(m_row["start_s"])

            calib_start = float(df_static.loc[run_id, "start_s"]) if run_id in df_static.index else 0.0
            time_gap = outage_start - calib_start
            flag = str(df_static.loc[run_id, "flag"]) if run_id in df_static.index else ""

            records.append({
                "outage_id": oid,
                "run_id": run_id,
                "outage_s": d,
                "bare_m": err_b,
                "corr_m": err_c,
                "diff_m": diff,
                "pct_diff": (diff / err_b) * 100.0 if err_b > 0 else 0.0,
                "calib_flag": flag,
                "time_gap_to_outage_s": time_gap,
            })

    return pd.DataFrame(records)


def main():
    print("=" * 80)
    print("STEP 17 DIAGNOSTIC: CALIBRATED-TIER REVERSAL ANALYSIS")
    print("=" * 80)

    # 1. Trajectory error curves
    print("\n--- 1. FULL TRAJECTORY ERROR CURVES (MEDIAN VS OUTAGE TIME) ---")
    curves = compute_trajectory_error_curves()
    for d, data in curves.items():
        tf = data["time_fractions"]
        mb = data["median_bare"]
        mc = data["median_corr"]
        print(f"\nOutage Duration: {d}s (N={data['n']} calibrated instances)")
        print(f"  Crossover point: {data['crossover_sec']:.1f}s ({data['crossover_frac']*100:.0f}% of outage)")
        idx_sample = [0, 5, 10, 15, 20]
        for idx in idx_sample:
            pct = int(tf[idx] * 100)
            t_s = tf[idx] * d
            delta = mc[idx] - mb[idx]
            status = "BETTER" if delta < 0 else "WORSE"
            print(f"    t={t_s:5.1f}s ({pct:3d}%): Bare={mb[idx]:8.1f}m | Corr={mc[idx]:8.1f}m | Delta={delta:+8.1f}m [{status}]")

    # 2. Gyro bias stability
    print("\n--- 2. GYRO BIAS STABILITY ACROSS MULTIPLE STILLNESS INTERVALS ---")
    stability = check_gyro_bias_stability()
    drift_df = stability["drift_df"]
    print(f"Calibrated runs evaluated: {stability['total_calibrated_runs']}")
    print(f"Runs with >=2 distinct quiet intervals: {stability['runs_with_multi_stops']}")
    if not drift_df.empty:
        print("\nMulti-stop drift statistics:")
        print(f"  Median time separation between stops: {drift_df['time_sep_s'].median():.1f}s ({drift_df['time_sep_s'].median()/60.0:.1f} min)")
        print(f"  Median gyro bias shift: {drift_df['gyro_shift_norm_rad_s'].median():.6f} rad/s ({drift_df['gyro_shift_norm_deg_s'].median():.3f} deg/s)")
        print(f"  Max gyro bias shift:    {drift_df['gyro_shift_norm_rad_s'].max():.6f} rad/s ({drift_df['gyro_shift_norm_deg_s'].max():.3f} deg/s)")
        print(f"  Median tilt shift:      pitch={drift_df['pitch_shift_deg'].abs().median():.2f} deg, roll={drift_df['roll_shift_deg'].abs().median():.2f} deg")
        print("\nIndividual multi-stop runs:")
        print(drift_df[["run_id", "time_sep_s", "gyro_shift_norm_deg_s", "pitch_shift_deg", "roll_shift_deg"]].to_string(index=False))

    # 3. Dynamic tilt & gravity leakage
    print("\n--- 3. DYNAMIC TILT & HORIZONTAL GRAVITY LEAKAGE DURING OUTAGES ---")
    leak_df = check_accelerometer_and_gravity_leakage()
    for d in [60, 180]:
        sub = leak_df[leak_df["outage_s"] == d]
        print(f"\nDuration {d}s (N={len(sub)}):")
        print(f"  Mean tilt deviation from initial leveling: {sub['mean_tilt_dev_deg'].mean():.2f} deg (max: {sub['max_tilt_dev_deg'].max():.2f} deg)")
        print(f"  Mean horizontal gravity leakage:          {sub['a_leak_mean_m_s2'].mean():.4f} m/s^2")
        print(f"  Theoretical position drift from leakage:  {sub['theoretical_leakage_drift_m'].median():.1f}m (median), {sub['theoretical_leakage_drift_m'].mean():.1f}m (mean)")

    # 4. Outliers & Culprits Analysis
    print("\n--- 4. OUTLIER & CULPRIT RUNS ANALYSIS ---")
    outlier_df = analyze_outlier_runs()
    for d in [60, 180]:
        sub = outlier_df[outlier_df["outage_s"] == d]
        print(f"\nTop 5 Degraded Outliers at {d}s:")
        print(sub.sort_values("diff_m", ascending=False).head(5)[["outage_id", "bare_m", "corr_m", "diff_m", "pct_diff", "time_gap_to_outage_s", "calib_flag"]].to_string(index=False))
        print(f"\nTop 5 Improved Outliers at {d}s:")
        print(sub.sort_values("diff_m", ascending=True).head(5)[["outage_id", "bare_m", "corr_m", "diff_m", "pct_diff", "time_gap_to_outage_s", "calib_flag"]].to_string(index=False))

    # 5. Suspect runs ablation
    print("\n--- 5. ABLATION ANALYSIS ON SUSPECT RUNS (pair_S2 & pair_Vta17) ---")
    abl_df = run_ablation_suspect_runs()
    print(abl_df.to_string(index=False))


if __name__ == "__main__":
    main()
