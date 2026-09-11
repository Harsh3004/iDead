import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/_cpp_replay_cache/pair_S3c__120s__0.csv", comment="#")
t = df["timestamp_s"].values
dt = np.diff(t)

# Integrate gx, gy, gz
int_gx = np.cumsum(df["gx"].values[:-1] * dt)
int_gy = np.cumsum(df["gy"].values[:-1] * dt)
int_gz = np.cumsum(df["gz"].values[:-1] * dt)

print("Total integrated rotation over 120s:")
print(f"  gx integrated: {np.degrees(int_gx[-1]):.2f}°")
print(f"  gy integrated: {np.degrees(int_gy[-1]):.2f}°")
print(f"  gz integrated: {np.degrees(int_gz[-1]):.2f}°")

# Let's check ground truth heading change
manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
df_parquet = pd.read_parquet(f"data/processed/outages/{row['split']}/pair_S3c__120s__0.parquet")
window = OutageWindow(
    run_id=row["run_or_pair_id"],
    source_side=row["source_side"],
    start_s=float(row["start_s"]),
    duration_s=float(row["outage_s"]),
    end_s=float(row["end_s"]),
)
gt = ground_truth_trajectory(df_parquet, window)
gt_hdg = gt["heading_deg"].values
gt_hdg_diff = np.unwrap(np.radians(gt_hdg))
total_gt_turn = np.degrees(gt_hdg_diff[-1] - gt_hdg_diff[0])
print(f"  Ground truth heading change: {total_gt_turn:.2f}°")

