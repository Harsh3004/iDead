"""Diagnose what last_f_body[0] actually measures across runs."""

import numpy as np
import pandas as pd
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_normalize, quat_multiply, quat_rotate, quat_to_rot_matrix

def parse_header(line):
    import re
    m = re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line)
    return {k: float(v) for k, v in m}

def from_euler_enu(yaw_deg, pitch_rad, roll_rad):
    half_yaw = 0.5 * np.radians(yaw_deg)
    q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, -np.sin(half_yaw)])
    half_pitch = 0.5 * pitch_rad
    q_pitch = np.array([np.cos(half_pitch), np.sin(half_pitch), 0.0, 0.0])
    half_roll = 0.5 * roll_rad
    q_roll = np.array([np.cos(half_roll), 0.0, np.sin(half_roll), 0.0])
    return quat_normalize(quat_multiply(quat_multiply(q_yaw, q_pitch), q_roll))

att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#").set_index("run_id")

sample_runs = [
    "pair_S1__180s__0",
    "pair_Vtb1__180s__0",
    "pair_S3b__180s__0",
    "pair_Vta17__180s__0",
    "pair_Vw4__30s__0",
    "pair_S3c__180s__0", # reference turn case
]

print(f"{'Outage ID':20s} | {'Roll (deg)':10s} | {'Pitch (deg)':10s} | {'Mean Raw fx':12s} | {'Std Raw fx':12s} | {'g*sin(roll)':12s} | {'True Dyn Alat':14s}")
print("-" * 105)

for oid in sample_runs:
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{oid}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    
    run_id = oid.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None
    
    roll_deg = np.degrees(att["roll_rad"]) if att is not None and pd.notna(att.get("roll_rad")) else 0.0
    pitch_deg = np.degrees(att["pitch_rad"]) if att is not None and pd.notna(att.get("pitch_rad")) else 0.0
    
    fx = df_imu["ax"].values
    mean_fx = np.mean(fx)
    std_fx = np.std(fx)
    
    g_roll = 9.80665 * np.sin(np.radians(roll_deg))
    
    # Let's run a simple strapdown to get true dynamic acceleration in body frame:
    # a_dynamic_body = f_body - R^T * [0, 0, 9.80665]
    # For roughly constant attitude:
    dyn_alat = np.abs(fx - g_roll)
    mean_dyn_alat = np.mean(dyn_alat)
    
    print(f"{oid:20s} | {roll_deg:10.2f} | {pitch_deg:10.2f} | {mean_fx:12.3f} | {std_fx:12.3f} | {g_roll:12.3f} | {mean_dyn_alat:14.3f}")
