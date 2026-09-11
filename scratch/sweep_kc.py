"""Test different k_c values and yaw-term inclusion across representative instances."""

import numpy as np
import pandas as pd
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_normalize, quat_multiply
from dataeval.harness.metrics import compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
import re

def parse_header(line):
    m = re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line)
    return {k: float(v) for k, v in m}

att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#").set_index("run_id")
manifest = pd.read_csv("data/processed/outages/manifest.csv").set_index("outage_id")

def from_euler_enu(yaw_deg, pitch_rad, roll_rad):
    half_yaw = 0.5 * np.radians(yaw_deg)
    q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, -np.sin(half_yaw)])
    half_pitch = 0.5 * pitch_rad
    q_pitch = np.array([np.cos(half_pitch), np.sin(half_pitch), 0.0, 0.0])
    half_roll = 0.5 * roll_rad
    q_roll = np.array([np.cos(half_roll), 0.0, np.sin(half_roll), 0.0])
    return quat_normalize(quat_multiply(quat_multiply(q_yaw, q_pitch), q_roll))

class KinematicEkf(ErrorStateEkf):
    def __init__(self, config=None, k_c=3.0, k_yaw=0.0):
        super().__init__(config)
        self.k_c = k_c
        self.k_yaw = k_yaw
        self.recorded_scales = []

    def compute_nhc_sigma_lat(self) -> float:
        dt_init = max(0.0, self.t - self.t0)
        settle_extra = 0.0
        if self.cfg.nhc_settle_duration_s > 1e-6:
            settle_extra = self.cfg.nhc_settle_sigma_extra * np.exp(-dt_init / self.cfg.nhc_settle_duration_s)

        speed = self.get_speed_ms()
        w_yaw = abs(self.last_omega_body[2] - self.b_gyro[2])
        a_c = speed * w_yaw

        term_c = self.k_c * a_c
        term_yaw = self.k_yaw * w_yaw
        curv_scale = np.sqrt(1.0 + term_c * term_c + term_yaw * term_yaw)
        self.recorded_scales.append(curv_scale)

        return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

def run_sim(outage_id, k_c, k_yaw):
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{outage_id}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    
    run_id = outage_id.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None
    
    cfg = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0)
    ekf = KinematicEkf(config=cfg, k_c=k_c, k_yaw=k_yaw)
    
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
    
    mean_scale = np.mean(ekf.recorded_scales) if ekf.recorded_scales else 1.0
    max_scale = np.max(ekf.recorded_scales) if ekf.recorded_scales else 1.0
    return metrics.final_pos_error_m, mean_scale, max_scale

test_instances = [
    # Highway turn target (pair_S3c)
    "pair_S3c__180s__0",
    "pair_S3c__120s__0",
    "pair_S3c__60s__0",
    "pair_S3c__30s__0",
    "pair_S3c__10s__0",
    # Regressed straight-road instances from Step 21
    "pair_S1__180s__0",
    "pair_S3b__180s__0",
    "pair_Vta17__180s__0",
    "pair_Vta1a__60s__0",
    "pair_Vtb1__180s__0",
    # 10s instance
    "pair_Vw16b__10s__0",
]

# Load official v1 and v2 errors from leaderboard
df_lb = pd.read_csv("results/leaderboard.csv")
v1_map = df_lb[df_lb["config"] == "ekf_zupt_nhc_v1"].set_index("outage_id")["final_pos_error_m"].to_dict()
v2_map = df_lb[df_lb["config"] == "ekf_zupt_nhc_v2"].set_index("outage_id")["final_pos_error_m"].to_dict()

for k_c in [1.0, 2.0, 3.0, 5.0]:
    for k_yaw in [0.0, 2.0]:
        print(f"\n=================== EVALUATION: k_c={k_c:.1f}, k_yaw={k_yaw:.1f} ===================")
        print(f"{'Outage ID':20s} | {'v1 (Old)':10s} | {'v2 (Step20)':12s} | {'Kinematic':10s} | {'Delta vs v1':12s} | {'Delta vs v2':12s} | {'MeanScale':10s} | {'MaxScale':10s}")
        print("-" * 105)
        for oid in test_instances:
            err, m_sc, mx_sc = run_sim(oid, k_c, k_yaw)
            e_v1 = v1_map.get(oid, 0.0)
            e_v2 = v2_map.get(oid, 0.0)
            d_v1 = err - e_v1
            d_v2 = err - e_v2
            print(f"{oid:20s} | {e_v1:8.2f}m  | {e_v2:8.2f}m    | {err:8.2f}m  | {d_v1:+8.2f}m    | {d_v2:+8.2f}m    | {m_sc:8.2f}x  | {mx_sc:8.2f}x")
