"""Compare k_yaw=0 vs k_yaw=2 vs v1/v2 across a 50-instance diverse sample."""

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
    def __init__(self, config=None, k_c=2.0, k_yaw=0.0):
        super().__init__(config)
        self.k_c = k_c
        self.k_yaw = k_yaw

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

        return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

def run_instance(outage_id, k_c, k_yaw):
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
    return metrics.final_pos_error_m

df_lb = pd.read_csv("results/leaderboard.csv")
v1_map = df_lb[df_lb["config"] == "ekf_zupt_nhc_v1"].set_index("outage_id")["final_pos_error_m"].to_dict()
v2_map = df_lb[df_lb["config"] == "ekf_zupt_nhc_v2"].set_index("outage_id")["final_pos_error_m"].to_dict()

# Select 50 instances: all pair_S3c instances (5), top 25 v2 regressed, top 10 v2 improved, 10 random
sample_oids = [oid for oid in v1_map.keys() if "pair_S3c" in oid]
regressed_oids = [oid for oid, diff in sorted({oid: v2_map[oid] - v1_map[oid] for oid in v1_map}.items(), key=lambda x: x[1], reverse=True)[:25]]
improved_oids = [oid for oid, diff in sorted({oid: v2_map[oid] - v1_map[oid] for oid in v1_map}.items(), key=lambda x: x[1])[:10]]
test_sample = list(dict.fromkeys(sample_oids + regressed_oids + improved_oids))[:50]

print(f"Testing {len(test_sample)} diverse instances across configurations...")

records = []
for oid in test_sample:
    e_v1 = v1_map[oid]
    e_v2 = v2_map[oid]
    e_kc2_y0 = run_instance(oid, k_c=2.0, k_yaw=0.0)
    e_kc2_y2 = run_instance(oid, k_c=2.0, k_yaw=2.0)
    e_kc1_y0 = run_instance(oid, k_c=1.0, k_yaw=0.0)
    
    records.append({
        "outage_id": oid,
        "e_v1": e_v1,
        "e_v2": e_v2,
        "e_kc2_y0": e_kc2_y0,
        "e_kc2_y2": e_kc2_y2,
        "e_kc1_y0": e_kc1_y0,
    })

res = pd.DataFrame(records)

print("\n--- Comparative Metrics on 50-Instance Sample ---")
print(f"Median Error:")
print(f"  v1 (Old Baseline):         {res['e_v1'].median():.2f}m")
print(f"  v2 (Step 20 Raw Accel):    {res['e_v2'].median():.2f}m")
print(f"  Kinematic k_c=2.0, y=0.0:  {res['e_kc2_y0'].median():.2f}m")
print(f"  Kinematic k_c=2.0, y=2.0:  {res['e_kc2_y2'].median():.2f}m")
print(f"  Kinematic k_c=1.0, y=0.0:  {res['e_kc1_y0'].median():.2f}m")

# Win rate vs v2:
win_kc2_y0_vs_v2 = (res['e_kc2_y0'] < res['e_v2'] - 0.1).sum()
win_kc2_y2_vs_v2 = (res['e_kc2_y2'] < res['e_v2'] - 0.1).sum()
win_kc1_y0_vs_v2 = (res['e_kc1_y0'] < res['e_v2'] - 0.1).sum()

print(f"\nWins vs v2 (on these 50 instances):")
print(f"  k_c=2.0, y=0.0 beats v2: {win_kc2_y0_vs_v2} / {len(res)} ({win_kc2_y0_vs_v2/len(res)*100:.1f}%)")
print(f"  k_c=2.0, y=2.0 beats v2: {win_kc2_y2_vs_v2} / {len(res)} ({win_kc2_y2_vs_v2/len(res)*100:.1f}%)")
print(f"  k_c=1.0, y=0.0 beats v2: {win_kc1_y0_vs_v2} / {len(res)} ({win_kc1_y0_vs_v2/len(res)*100:.1f}%)")

# Compare k_yaw=0 vs k_yaw=2 directly:
diff_y = res['e_kc2_y0'] - res['e_kc2_y2']
y0_better = (diff_y < -0.1).sum()
y2_better = (diff_y > 0.1).sum()
y_equal = (diff_y.abs() <= 0.1).sum()
print(f"\nDirect comparison (k_c=2.0, y=0 vs y=2):")
print(f"  y=0 better: {y0_better}, y=2 better: {y2_better}, identical within 0.1m: {y_equal}")
