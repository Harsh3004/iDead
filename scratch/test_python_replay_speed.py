import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_normalize, quat_multiply
from dataeval.harness.metrics import compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

def parse_header(line):
    # initial_state: t0=4476.100000,lat0=52.40249600,lon0=-1.56518700,alt0=168.870,speed_ms=9.5600,heading_deg=179.9900,ax0=0.0000,ay0=0.0000,az0=9.8066
    m = re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line)
    return {k: float(v) for k, v in m}

def load_attitude(attitude_csv=Path("results/module_b_initial_attitude.csv")):
    df = pd.read_csv(attitude_csv, comment="#")
    return df.set_index("run_id")

att_df = load_attitude()

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
    
    # Attitude
    run_id = outage_id.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None
    
    ekf = ErrorStateEkf(config)
    
    # Build initial quaternion
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
    
    t_start = time.perf_counter()
    preds = []
    for _, row in df_imu.iterrows():
        t = row["timestamp_s"]
        f_b = np.array([row["ax"], row["ay"], row["az"]])
        omega_b = np.array([row["gx"], row["gy"], row["gz"]])
        
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
    elapsed = time.perf_counter() - t_start
    
    # Ground truth
    manifest = pd.read_csv("data/processed/outages/manifest.csv").set_index("outage_id")
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
    return metrics.final_pos_error_m, elapsed

# Test 4 variants on pair_S1__180s__0 (worst regression)
cfg_v1 = EkfConfig(nhc_settle_sigma_extra=0.0, nhc_curv_lat_coeff=0.0, nhc_curv_yaw_coeff=0.0)
cfg_settle = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0, nhc_curv_lat_coeff=0.0, nhc_curv_yaw_coeff=0.0)
cfg_curv = EkfConfig(nhc_settle_sigma_extra=0.0, nhc_curv_lat_coeff=5.0, nhc_curv_yaw_coeff=2.0)
cfg_v2 = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0, nhc_curv_lat_coeff=5.0, nhc_curv_yaw_coeff=2.0)

for name, cfg in [("v1 (Baseline)", cfg_v1), ("settle_only", cfg_settle), ("curv_only", cfg_curv), ("v2 (Both)", cfg_v2)]:
    err, el = run_instance_ekf("pair_S1__180s__0", cfg)
    print(f"pair_S1__180s__0 | {name:15s} -> Error: {err:8.2f}m (ran in {el:.2f}s)")

