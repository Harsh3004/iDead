import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]
split = row["split"]

df_parquet = pd.read_parquet(f"data/processed/outages/{split}/pair_S3c__120s__0.parquet")
window = OutageWindow(
    run_id=row["run_or_pair_id"], source_side=row["source_side"],
    start_s=float(row["start_s"]), duration_s=float(row["outage_s"]), end_s=float(row["end_s"])
)
gt = ground_truth_trajectory(df_parquet, window)

pred_v2 = load_cpp_prediction(Path("data/processed/cpp_predictions_ekf_v2/pair_S3c__120s__0.csv"))
pred_v3 = load_cpp_prediction(Path("data/processed/cpp_predictions_ekf_v3/pair_S3c__120s__0.csv"))

t = pred_v3["timestamp_s"].values
t_rel = t - t[0]

# Local ENU
lat0, lon0 = np.radians(gt["latitude_deg"].iloc[0]), np.radians(gt["longitude_deg"].iloc[0])
kEarthRadius = 6371000.0
def to_enu(lats, lons):
    dlat = np.radians(lats) - lat0
    dlon = np.radians(lons) - lon0
    north = dlat * kEarthRadius
    east = dlon * (kEarthRadius * np.cos(lat0))
    return east, north

gt_e, gt_n = to_enu(gt["latitude_deg"], gt["longitude_deg"])
v2_e, v2_n = to_enu(pred_v2["latitude_deg"], pred_v2["longitude_deg"])
v3_e, v3_n = to_enu(pred_v3["latitude_deg"], pred_v3["longitude_deg"])

# Interpolate GT to pred timestamps
gt_e_interp = np.interp(t, gt["timestamp_s"], gt_e)
gt_n_interp = np.interp(t, gt["timestamp_s"], gt_n)

# Compute along-track and cross-track errors
# Compute tangent of GT at each point
gt_vel_e = np.gradient(gt_e_interp, t)
gt_vel_n = np.gradient(gt_n_interp, t)
gt_speed = np.hypot(gt_vel_e, gt_vel_n)
gt_speed[gt_speed < 0.1] = 0.1
# Unit tangent (along) and normal (cross, 90 deg right)
tan_e = gt_vel_e / gt_speed
tan_n = gt_vel_n / gt_speed
norm_e = tan_n
norm_n = -tan_e

# Errors for v2
err_e_v2 = v2_e - gt_e_interp
err_n_v2 = v2_n - gt_n_interp
along_v2 = err_e_v2 * tan_e + err_n_v2 * tan_n
cross_v2 = err_e_v2 * norm_e + err_n_v2 * norm_n
total_v2 = np.hypot(err_e_v2, err_n_v2)

# Errors for v3
err_e_v3 = v3_e - gt_e_interp
err_n_v3 = v3_n - gt_n_interp
along_v3 = err_e_v3 * tan_e + err_n_v3 * tan_n
cross_v3 = err_e_v3 * norm_e + err_n_v3 * norm_n
total_v3 = np.hypot(err_e_v3, err_n_v3)

print("Trajectory Error Evolution for pair_S3c__120s__0:")
print("  t(s) |  GT dist (m) |  v2 TotErr |  v3 TotErr |  v2 Along  |  v3 Along  |  v2 Cross  |  v3 Cross  | v2-v3 Delta")
print("-------+--------------+------------+------------+------------+------------+------------+------------+-------------")
for step_t in range(0, 121, 10):
    idx = np.argmin(np.abs(t_rel - step_t))
    gt_dist = np.hypot(gt_e_interp[idx], gt_n_interp[idx])
    print(f"  {step_t:4d} |  {gt_dist:10.1f}m |  {total_v2[idx]:8.1f}m |  {total_v3[idx]:8.1f}m |  {along_v2[idx]:8.1f}m |  {along_v3[idx]:8.1f}m |  {cross_v2[idx]:8.1f}m |  {cross_v3[idx]:8.1f}m | {total_v3[idx]-total_v2[idx]:+10.1f}m")

# Check end points
print("\nFinal Position at t=120s:")
print(f"  GT: East={gt_e_interp[-1]:.1f}m, North={gt_n_interp[-1]:.1f}m (Distance traveled={np.hypot(gt_e_interp[-1], gt_n_interp[-1]):.1f}m)")
print(f"  v2: East={v2_e.iloc[-1]:.1f}m, North={v2_n.iloc[-1]:.1f}m -> Error={total_v2.iloc[-1]:.1f}m (Along={along_v2.iloc[-1]:.1f}m, Cross={cross_v2.iloc[-1]:.1f}m)")
print(f"  v3: East={v3_e.iloc[-1]:.1f}m, North={v3_n.iloc[-1]:.1f}m -> Error={total_v3.iloc[-1]:.1f}m (Along={along_v3.iloc[-1]:.1f}m, Cross={cross_v3.iloc[-1]:.1f}m)")

