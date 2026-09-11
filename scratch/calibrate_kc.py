"""Analyze kinematic centripetal acceleration on pair_S3c and straight-road runs to calibrate k_c."""

import numpy as np
import pandas as pd
from pathlib import Path
import re

def parse_header(line):
    m = re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line)
    return {k: float(v) for k, v in m}

att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#").set_index("run_id")

runs = [
    # Highway turn target
    ("pair_S3c__180s__0", "Highway Curve (S3c)"),
    ("pair_S3c__120s__0", "Highway Curve (S3c)"),
    ("pair_S3c__30s__0", "Highway Curve (S3c)"),
    # Straight-road regressions from Step 21
    ("pair_S1__180s__0", "Straight Highway (S1)"),
    ("pair_S3b__180s__0", "Straight Highway (S3b)"),
    ("pair_Vta17__180s__0", "Suburban Street (Vta17)"),
    ("pair_Vta1a__60s__0", "Urban Arterial (Vta1a)"),
    ("pair_Vtb1__180s__0", "Urban Street (Vtb1)"),
    ("pair_Vw4__30s__0", "Mixed Road (Vw4)"),
]

print(f"{'Outage ID':20s} | {'Type':24s} | {'Speed(m/s)':10s} | {'Mean |gz|':10s} | {'Max |gz|':10s} | {'Mean a_c':10s} | {'p95 a_c':10s} | {'Max a_c':10s} | {'Old Mean ax':12s}")
print("-" * 130)

for oid, desc in runs:
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{oid}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    
    run_id = oid.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None
    
    gb_z = float(att["gyro_bias_z_rad_s"]) if att is not None and pd.notna(att.get("gyro_bias_z_rad_s")) else 0.0
    
    gz_corrected = np.abs(df_imu["gz"].values - gb_z)
    ax_raw = np.abs(df_imu["ax"].values)
    
    # Let's approximate speed as initial speed plus integrated longitudinal accel or speed_ms
    v0 = float(init.get("speed_ms", 15.0))
    # Or rough speed from initial
    a_c = v0 * gz_corrected
    
    print(f"{oid:20s} | {desc:24s} | {v0:10.2f} | {np.mean(gz_corrected):10.4f} | {np.max(gz_corrected):10.4f} | {np.mean(a_c):10.3f} | {np.percentile(a_c, 95):10.3f} | {np.max(a_c):10.3f} | {np.mean(ax_raw):12.3f}")
