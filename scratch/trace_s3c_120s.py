import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig
from dataeval.harness.run_corrected_replay import load_module_b_tiers

# 1. Load manifest info for pair_S3c__120s__0
manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]

run_id = row["run_or_pair_id"]
source_side = row["source_side"]
start_s = float(row["start_s"])
end_s = float(row["end_s"])
outage_s = float(row["outage_s"])
split = row["split"]

parquet_path = Path(f"data/processed/outages/{split}/pair_S3c__120s__0.parquet")
df_parquet = pd.read_parquet(parquet_path)

window = OutageWindow(
    run_id=run_id,
    source_side=source_side,
    start_s=start_s,
    duration_s=outage_s,
    end_s=end_s,
)
gt_df = ground_truth_trajectory(df_parquet, window)

# 2. Load v1, v2, v3 predictions
pred_v1 = load_cpp_prediction(Path("data/processed/cpp_predictions_ekf/pair_S3c__120s__0.csv"))
pred_v2 = load_cpp_prediction(Path("data/processed/cpp_predictions_ekf_v2/pair_S3c__120s__0.csv"))
pred_v3 = load_cpp_prediction(Path("data/processed/cpp_predictions_ekf_v3/pair_S3c__120s__0.csv"))

print(f"GT points: {len(gt_df)}, Pred points: v1={len(pred_v1)}, v2={len(pred_v2)}, v3={len(pred_v3)}")

# Interpolate GT to prediction timestamps to get continuous error over time
t_pred = pred_v3["timestamp_s"].values
t_rel = t_pred - t_pred[0]

gt_lat = np.interp(t_pred, gt_df["timestamp_s"], gt_df["latitude_deg"])
gt_lon = np.interp(t_pred, gt_df["timestamp_s"], gt_df["longitude_deg"])
gt_speed = np.interp(t_pred, gt_df["timestamp_s"], gt_df["speed_ms"])
gt_heading = np.interp(t_pred, gt_df["timestamp_s"], gt_df["heading_deg"])

# Convert to local ENU coordinates relative to origin (first GT point)
kEarthRadius = 6371000.0
lat0, lon0 = np.radians(gt_lat[0]), np.radians(gt_lon[0])

def to_enu(lats, lons):
    dlat = np.radians(lats) - lat0
    dlon = np.radians(lons) - lon0
    north = dlat * kEarthRadius
    east = dlon * (kEarthRadius * np.cos(lat0))
    return east, north

gt_e, gt_n = to_enu(gt_lat, gt_lon)
v1_e, v1_n = to_enu(pred_v1["latitude_deg"], pred_v1["longitude_deg"])
v2_e, v2_n = to_enu(pred_v2["latitude_deg"], pred_v2["longitude_deg"])
v3_e, v3_n = to_enu(pred_v3["latitude_deg"], pred_v3["longitude_deg"])

err_v1 = np.hypot(v1_e - gt_e, v1_n - gt_n)
err_v2 = np.hypot(v2_e - gt_e, v2_n - gt_n)
err_v3 = np.hypot(v3_e - gt_e, v3_n - gt_n)

# Check errors at 10s intervals
print("\n--- Position Error Over Time (m) ---")
print("  t(s) |     GT Speed |       v1 Err |       v2 Err |       v3 Err |  Diff(v3-v2)")
print("-------+--------------+--------------+--------------+--------------+-------------")
for step_t in range(0, int(outage_s) + 1, 10):
    idx = np.argmin(np.abs(t_rel - step_t))
    print(f"  {step_t:4d} |  {gt_speed[idx]:6.1f} m/s |  {err_v1[idx]:8.1f} m |  {err_v2[idx]:8.1f} m |  {err_v3[idx]:8.1f} m |  {err_v3[idx] - err_v2[idx]:+8.1f} m")

# Speed errors
err_spd_v1 = pred_v1["speed_ms"].values - gt_speed
err_spd_v2 = pred_v2["speed_ms"].values - gt_speed
err_spd_v3 = pred_v3["speed_ms"].values - gt_speed

print("\n--- Speed Error Over Time (m/s) ---")
print("  t(s) |  v1 Spd Err  |  v2 Spd Err  |  v3 Spd Err  |  v3 Spd (est)")
print("-------+--------------+--------------+--------------+--------------")
for step_t in range(0, int(outage_s) + 1, 10):
    idx = np.argmin(np.abs(t_rel - step_t))
    print(f"  {step_t:4d} |  {err_spd_v1[idx]:+8.2f} m/s |  {err_spd_v2[idx]:+8.2f} m/s |  {err_spd_v3[idx]:+8.2f} m/s |  {pred_v3['speed_ms'].values[idx]:6.2f} m/s")

# Heading errors (circular diff)
def angle_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

err_hdg_v1 = angle_diff(pred_v1["heading_deg"].values, gt_heading)
err_hdg_v2 = angle_diff(pred_v2["heading_deg"].values, gt_heading)
err_hdg_v3 = angle_diff(pred_v3["heading_deg"].values, gt_heading)

print("\n--- Heading Error Over Time (deg) ---")
print("  t(s) |  v1 Hdg Err  |  v2 Hdg Err  |  v3 Hdg Err  |  Diff(v3-v2)")
print("-------+--------------+--------------+--------------+-------------")
for step_t in range(0, int(outage_s) + 1, 10):
    idx = np.argmin(np.abs(t_rel - step_t))
    print(f"  {step_t:4d} |  {err_hdg_v1[idx]:+8.1f}° |  {err_hdg_v2[idx]:+8.1f}° |  {err_hdg_v3[idx]:+8.1f}° |  {err_hdg_v3[idx] - err_hdg_v2[idx]:+8.1f}°")

