import pandas as pd
import numpy as np

p2 = pd.read_csv("data/processed/cpp_predictions_ekf_v2/pair_S3c__120s__0.csv")
p3 = pd.read_csv("data/processed/cpp_predictions_ekf_v3/pair_S3c__120s__0.csv")

manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
df_parquet = pd.read_parquet(f"data/processed/outages/{row['split']}/pair_S3c__120s__0.parquet")
window = OutageWindow(run_id=row["run_or_pair_id"], source_side=row["source_side"], start_s=float(row["start_s"]), duration_s=float(row["outage_s"]), end_s=float(row["end_s"]))
gt = ground_truth_trajectory(df_parquet, window)

lat0, lon0 = np.radians(gt["latitude_deg"].iloc[0]), np.radians(gt["longitude_deg"].iloc[0])
kEarth = 6371000.0

def to_enu(lats, lons):
    dlat = np.radians(lats) - lat0
    dlon = np.radians(lons) - lon0
    return dlon * (kEarth * np.cos(lat0)), dlat * kEarth

gt_e, gt_n = to_enu(gt["latitude_deg"], gt["longitude_deg"])
p2_e, p2_n = to_enu(p2["latitude_deg"], p2["longitude_deg"])
p3_e, p3_n = to_enu(p3["latitude_deg"], p3["longitude_deg"])

t_p = p3["timestamp_s"].values - p3["timestamp_s"].iloc[0]

print("Trajectory Coordinates (East, North) in meters every 20s:")
print(f"{'t(s)':5s} | {'GT (East, North)':22s} | {'v2 (East, North)':22s} | {'v3 (East, North)':22s} | {'v2 Err':8s} | {'v3 Err':8s}")
print("-" * 95)
for step in range(0, 121, 20):
    idx_p = np.argmin(np.abs(t_p - step))
    # interpolate GT
    gt_e_val = np.interp(p3["timestamp_s"].iloc[idx_p], gt["timestamp_s"], gt_e)
    gt_n_val = np.interp(p3["timestamp_s"].iloc[idx_p], gt["timestamp_s"], gt_n)
    
    e2, n2 = p2_e.iloc[idx_p], p2_n.iloc[idx_p]
    e3, n3 = p3_e.iloc[idx_p], p3_n.iloc[idx_p]
    
    err2 = np.hypot(e2 - gt_e_val, n2 - gt_n_val)
    err3 = np.hypot(e3 - gt_e_val, n3 - gt_n_val)
    
    print(f"{step:4d}s | ({gt_e_val:+7.1f}, {gt_n_val:+7.1f}) | ({e2:+7.1f}, {n2:+7.1f}) | ({e3:+7.1f}, {n3:+7.1f}) | {err2:7.1f}m | {err3:7.1f}m")

# Check strapdown INS baseline (no EKF)
p_ins = pd.read_csv("data/processed/cpp_predictions/pair_S3c__120s__0.csv")
ins_e, ins_n = to_enu(p_ins["latitude_deg"], p_ins["longitude_deg"])
err_ins = np.hypot(ins_e.iloc[-1] - gt_e.iloc[-1], ins_n.iloc[-1] - gt_n.iloc[-1])
print(f"\nBare Strapdown INS final error: {err_ins:.1f}m")

# Check corrected strapdown INS baseline (Step 16)
p_corr = pd.read_csv("data/processed/cpp_predictions_corrected/pair_S3c__120s__0.csv")
corr_e, corr_n = to_enu(p_corr["latitude_deg"], p_corr["longitude_deg"])
err_corr = np.hypot(corr_e.iloc[-1] - gt_e.iloc[-1], corr_n.iloc[-1] - gt_n.iloc[-1])
print(f"Corrected Strapdown INS final error: {err_corr:.1f}m")
