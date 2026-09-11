"""Analyze contributions of a_lat vs w_yaw to curvature scale."""

import numpy as np
import pandas as pd
from pathlib import Path

sample_runs = [
    "pair_S1__180s__0",
    "pair_Vtb1__180s__0",
    "pair_S3b__180s__0",
    "pair_Vta17__180s__0",
    "pair_Vw4__30s__0",
    "pair_S3c__180s__0",
]

print(f"{'Outage ID':20s} | {'Mean |ax|':10s} | {'(5*ax)^2':10s} | {'Mean |gz|':10s} | {'(2*gz)^2':10s} | {'Lat Dominance %':16s}")
print("-" * 85)

for oid in sample_runs:
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{oid}.csv")
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    
    ax = np.abs(df_imu["ax"].values)
    gz = np.abs(df_imu["gz"].values)
    
    term_lat = (5.0 * ax) ** 2
    term_yaw = (2.0 * gz) ** 2
    
    mean_lat_sq = np.mean(term_lat)
    mean_yaw_sq = np.mean(term_yaw)
    
    lat_dom = mean_lat_sq / (mean_lat_sq + mean_yaw_sq + 1e-6) * 100.0
    
    print(f"{oid:20s} | {np.mean(ax):10.3f} | {mean_lat_sq:10.2f} | {np.mean(gz):10.3f} | {mean_yaw_sq:10.2f} | {lat_dom:15.1f}%")
