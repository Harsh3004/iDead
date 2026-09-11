"""Run the 4 controlled EKF variants on top regressed and improved instances."""

import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_normalize, quat_multiply
from dataeval.harness.metrics import compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

def parse_header(line):
    m = re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line)
    return {k: float(v) for k, v in m}

def load_attitude(attitude_csv=Path("results/module_b_initial_attitude.csv")):
    df = pd.read_csv(attitude_csv, comment="#")
    return df.set_index("run_id")

att_df = load_attitude()
manifest = pd.read_csv("data/processed/outages/manifest.csv").set_index("outage_id")

def from_euler_enu(yaw_deg, pitch_rad, roll_rad):
    half_yaw = 0.5 * np.radians(yaw_deg)
    q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, -np.sin(half_yaw)])
    half_pitch = 0.5 * pitch_rad
    q_pitch = np.array([np.cos(half_pitch), np.sin(half_pitch), 0.0, 0.0])
    half_roll = 0.5 * roll_rad
    q_roll = np.array([np.cos(half_roll), 0.0, np.sin(half_roll), 0.0])
    return quat_normalize(quat_multiply(quat_multiply(q_yaw, q_pitch), q_roll))

def run_instance_ekf(outage_id, config: EkfConfig):
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{outage_id}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    
    run_id = outage_id.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None
    
    ekf = ErrorStateEkf(config)
    
    if att is not None and pd.notna(att.get("q_w")):
        q0 = np.array([att["q_w"], att["q_x"], att["q_y"], att["q_z"]])
    elif att is not None and att.get("leveling_source") == "module_a_measured":
        q0 = from_euler_enu(init["heading_deg"], att["pitch_rad"], att["roll_rad"])
    else:
        half_psi = 0.5 * np.radians(init["heading_deg"])
        q0 = np.array([np.cos(half_psi), 0.0, 0.0, -np.sin(half_psi)])
        
    gyro_bias = np.zeros(3)
    if att is not None and att.get("gyro_bias_source") == "module_a_measured":
        gyro_bias = np.array([att["gyro_bias_x_rad_s"], att["gyro_bias_y_rad_s"], att["gyro_bias_z_rad_s"]])
        
    ekf.initialize(
        t0=init["t0"],
        lat0=init["lat0"],
        lon0=init["lon0"],
        alt0=init["alt0"],
        speed_ms=init["speed_ms"],
        heading_deg=init["heading_deg"],
        q0=q0,
        b_gyro0=gyro_bias,
    )
    
    preds = []
    # Collect statistics about curvature inflation during the run
    curv_scales = []
    a_lats = []
    w_yaws = []
    
    for _, row in df_imu.iterrows():
        t = row["timestamp_s"]
        f_b = np.array([row["ax"], row["ay"], row["az"]])
        omega_b = np.array([row["gx"], row["gy"], row["gz"]])
        
        # Track raw a_lat and w_yaw before predict
        a_lat = abs(f_b[0] - ekf.b_accel[0])
        w_yaw = abs(omega_b[2] - ekf.b_gyro[2])
        lat_term = config.nhc_curv_lat_coeff * a_lat
        yaw_term = config.nhc_curv_yaw_coeff * w_yaw
        curv_scale = np.sqrt(1.0 + lat_term * lat_term + yaw_term * yaw_term)
        curv_scales.append(curv_scale)
        a_lats.append(a_lat)
        w_yaws.append(w_yaw)
        
        ekf.predict(t, f_b, omega_b)
        if ekf.is_stationary():
            ekf.update_zupt()
        ekf.update_nhc()
        
        lat, lon, alt = ekf.get_position_lat_lon()
        preds.append({
            "timestamp_s": t,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "speed_ms": ekf.get_speed_ms(),
            "heading_deg": ekf.get_heading_deg(),
        })
        
    pred_df = pd.DataFrame(preds)
    man = manifest.loc[outage_id]
    outage_parquet = Path(f"data/processed/outages/{man['split']}/{outage_id}.parquet")
    df_parquet = pd.read_parquet(outage_parquet)
    window = OutageWindow(
        run_id=man["run_or_pair_id"],
        source_side=man["source_side"],
        start_s=man["start_s"],
        duration_s=float(man["outage_s"]),
        end_s=man["end_s"],
    )
    gt_df = ground_truth_trajectory(df_parquet, window)
    metrics = compute_outage_metrics(pred_df, gt_df)
    
    stats = {
        "mean_curv_scale": np.mean(curv_scales),
        "max_curv_scale": np.max(curv_scales),
        "mean_a_lat": np.mean(a_lats),
        "p95_a_lat": np.percentile(a_lats, 95),
        "max_a_lat": np.max(a_lats),
        "mean_w_yaw": np.mean(w_yaws),
        "p95_w_yaw": np.percentile(w_yaws, 95),
        "max_w_yaw": np.max(w_yaws),
    }
    return metrics.final_pos_error_m, stats

cfg_v1 = EkfConfig(nhc_settle_sigma_extra=0.0, nhc_curv_lat_coeff=0.0, nhc_curv_yaw_coeff=0.0)
cfg_settle = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0, nhc_curv_lat_coeff=0.0, nhc_curv_yaw_coeff=0.0)
cfg_curv = EkfConfig(nhc_settle_sigma_extra=0.0, nhc_curv_lat_coeff=5.0, nhc_curv_yaw_coeff=2.0)
cfg_v2 = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0, nhc_curv_lat_coeff=5.0, nhc_curv_yaw_coeff=2.0)

# Select test instances: top regressed, diverse durations, and some improved
test_instances = [
    # Top regressed
    "pair_S1__180s__0",
    "pair_Vtb1__180s__0",
    "pair_S2__120s__0",
    "pair_S3b__180s__0",
    "pair_Vta1a__180s__0",
    "pair_Vta1a__60s__0",
    "pair_Vta17__180s__0",
    "pair_Vta16__180s__0",
    "pair_Vta29__120s__0",
    "pair_Vta15__60s__0",
    "pair_Vta28__180s__0",
    "pair_Vtb1__120s__0",
    # 30s regressed
    "pair_S3b__30s__0",
    "pair_Vta1a__30s__0",
    "pair_Vw4__30s__0",
    # 10s regressed
    "pair_Vw10__10s__0",
    "pair_Vtb8__10s__0",
    # Key improved
    "pair_S3c__180s__0",
    "pair_S3c__120s__0",
    "pair_S4__120s__0",
    "pair_M__180s__0",
    "pair_Vw16b__10s__0",
]

print(f"{'Outage ID':20s} | {'Dur':4s} | {'v1 (Base)':10s} | {'SettleOnly':10s} | {'CurvOnly':10s} | {'v2 (Both)':10s} | {'Prim Driver':12s} | {'Mean a_lat':10s} | {'Max a_lat':10s} | {'Mean CurvScale':14s}")
print("-" * 125)

results = []
for oid in test_instances:
    dur = oid.split("__")[1]
    e_v1, s_v1 = run_instance_ekf(oid, cfg_v1)
    e_set, s_set = run_instance_ekf(oid, cfg_settle)
    e_curv, s_curv = run_instance_ekf(oid, cfg_curv)
    e_v2, s_v2 = run_instance_ekf(oid, cfg_v2)
    
    diff_set = e_set - e_v1
    diff_curv = e_curv - e_v1
    diff_v2 = e_v2 - e_v1
    
    # Determine primary driver
    if abs(diff_set) > abs(diff_curv) * 2:
        driver = "SETTLE"
    elif abs(diff_curv) > abs(diff_set) * 2:
        driver = "CURVATURE"
    elif abs(diff_v2) < 1.0:
        driver = "NEUTRAL"
    else:
        driver = "BOTH"
        
    print(f"{oid:20s} | {dur:4s} | {e_v1:8.2f}m  | {e_set:8.2f}m  | {e_curv:8.2f}m  | {e_v2:8.2f}m  | {driver:12s} | {s_v2['mean_a_lat']:8.3f}   | {s_v2['max_a_lat']:8.3f}  | {s_v2['mean_curv_scale']:10.2f}x")
    results.append({
        "outage_id": oid,
        "duration": dur,
        "e_v1": e_v1,
        "e_set": e_set,
        "e_curv": e_curv,
        "e_v2": e_v2,
        "driver": driver,
        **s_v2
    })
