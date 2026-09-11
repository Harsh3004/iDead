import pandas as pd
import numpy as np
from pathlib import Path

p_v1 = pd.read_csv("data/processed/cpp_predictions_ekf/pair_S3c__120s__0.csv")
p_v2 = pd.read_csv("data/processed/cpp_predictions_ekf_v2/pair_S3c__120s__0.csv")
p_v3 = pd.read_csv("data/processed/cpp_predictions_ekf_v3/pair_S3c__120s__0.csv")

manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
df_parquet = pd.read_parquet(f"data/processed/outages/{row['split']}/pair_S3c__120s__0.parquet")
window = OutageWindow(
    run_id=row["run_or_pair_id"], source_side=row["source_side"],
    start_s=float(row["start_s"]), duration_s=float(row["outage_s"]), end_s=float(row["end_s"])
)
gt = ground_truth_trajectory(df_parquet, window)

t_gt = gt["timestamp_s"].values
gt_lat = np.interp(p_v3["timestamp_s"], t_gt, gt["latitude_deg"])
gt_lon = np.interp(p_v3["timestamp_s"], t_gt, gt["longitude_deg"])
gt_speed = np.interp(p_v3["timestamp_s"], t_gt, gt["speed_ms"])
gt_hdg = np.interp(p_v3["timestamp_s"], t_gt, gt["heading_deg"])

kEarthRadius = 6371000.0
lat0, lon0 = np.radians(gt_lat[0]), np.radians(gt_lon[0])

def to_enu(lats, lons):
    dlat = np.radians(lats) - lat0
    dlon = np.radians(lons) - lon0
    north = dlat * kEarthRadius
    east = dlon * (kEarthRadius * np.cos(lat0))
    return east, north

gt_e, gt_n = to_enu(gt_lat, gt_lon)
e_v1, n_v1 = to_enu(p_v1["latitude_deg"], p_v1["longitude_deg"])
e_v2, n_v2 = to_enu(p_v2["latitude_deg"], p_v2["longitude_deg"])
e_v3, n_v3 = to_enu(p_v3["latitude_deg"], p_v3["longitude_deg"])

err_v1 = np.hypot(e_v1 - gt_e, n_v1 - gt_n)
err_v2 = np.hypot(e_v2 - gt_e, n_v2 - gt_n)
err_v3 = np.hypot(e_v3 - gt_e, n_v3 - gt_n)

t_rel = p_v3["timestamp_s"] - p_v3["timestamp_s"].iloc[0]

print("C++ Predictions for pair_S3c__120s__0 Every 10s:")
print("  t(s) |  GT Spd |  v2 Spd |  v3 Spd |  GT Hdg  |  v2 Hdg  |  v3 Hdg  |  v2 Err  |  v3 Err  | Delta(v3-v2)")
print("-------+---------+---------+---------+----------+----------+----------+----------+----------+-------------")
for step_t in range(0, 121, 10):
    idx = np.argmin(np.abs(t_rel - step_t))
    print(f"  {step_t:4d} |  {gt_speed[idx]:5.1f}  |  {p_v2['speed_ms'].iloc[idx]:5.1f}  |  {p_v3['speed_ms'].iloc[idx]:5.1f}  |  {gt_hdg[idx]:6.1f}° |  {p_v2['heading_deg'].iloc[idx]:6.1f}° |  {p_v3['heading_deg'].iloc[idx]:6.1f}° | {err_v2[idx]:7.1f}m | {err_v3[idx]:7.1f}m | {err_v3[idx]-err_v2[idx]:+9.1f}m")

