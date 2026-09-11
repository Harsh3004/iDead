import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]

parquet_path = Path(f"data/processed/outages/{row['split']}/pair_S3c__120s__0.parquet")
df_parquet = pd.read_parquet(parquet_path)

window = OutageWindow(
    run_id=row["run_or_pair_id"],
    source_side=row["source_side"],
    start_s=float(row["start_s"]),
    duration_s=float(row["outage_s"]),
    end_s=float(row["end_s"]),
)
gt_df = ground_truth_trajectory(df_parquet, window)

# Compute GT yaw rate from heading derivative
t = gt_df["timestamp_s"].values
hdg = gt_df["heading_deg"].values
speed = gt_df["speed_ms"].values

# unwrap heading for clean differentiation
hdg_unwrapped = np.unwrap(np.radians(hdg))
dt = np.diff(t)
w_yaw_gt = np.abs(np.diff(hdg_unwrapped) / dt)
a_c_gt = speed[:-1] * w_yaw_gt

print(f"GT Outage Summary for pair_S3c__120s__0 (Duration {len(gt_df)*0.1:.1f}s):")
print(f"  GT Speed: mean={np.mean(speed):.2f} m/s, min={np.min(speed):.2f}, max={np.max(speed):.2f}")
print(f"  GT Yaw Rate: mean={np.degrees(np.mean(w_yaw_gt)):.2f}°/s, max={np.degrees(np.max(w_yaw_gt)):.2f}°/s")
print(f"  GT a_c = v * w: mean={np.mean(a_c_gt):.3f} m/s², max={np.max(a_c_gt):.3f} m/s², 95th percentile={np.percentile(a_c_gt, 95):.3f} m/s²")

# Check where GT a_c is high
high_curve = np.where(a_c_gt > 0.5)[0]
print(f"  Number of samples with a_c > 0.5 m/s²: {len(high_curve)} / {len(a_c_gt)} ({len(high_curve)/len(a_c_gt)*100:.1f}%)")
high_curve_1 = np.where(a_c_gt > 1.0)[0]
print(f"  Number of samples with a_c > 1.0 m/s²: {len(high_curve_1)} / {len(a_c_gt)} ({len(high_curve_1)/len(a_c_gt)*100:.1f}%)")

# Sample GT values every 10s
print("\n--- GT Profile Every 10s ---")
print("  t_rel(s) |  GT Speed (m/s) |  GT Heading (deg) |  GT YawRate (deg/s) |  GT a_c (m/s²)")
for step_t in range(0, 121, 10):
    idx = np.argmin(np.abs((t - t[0]) - step_t))
    w = np.degrees(w_yaw_gt[min(idx, len(w_yaw_gt)-1)])
    ac = a_c_gt[min(idx, len(a_c_gt)-1)]
    print(f"  {step_t:8d} |  {speed[idx]:14.2f} |  {hdg[idx]:17.2f} |  {w:19.3f} |  {ac:14.3f}")

