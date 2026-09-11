import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

manifest = pd.read_csv("data/processed/outages/manifest.csv")
s3c_rows = manifest[manifest["run_or_pair_id"] == "pair_S3c"].sort_values("outage_s")

print("--- pair_S3c Outages Profile ---")
for _, row in s3c_rows.iterrows():
    oid = row["outage_id"]
    split = row["split"]
    dur = row["outage_s"]
    start = row["start_s"]
    end = row["end_s"]
    
    # GT
    df_parquet = pd.read_parquet(f"data/processed/outages/{split}/{oid}.parquet")
    window = OutageWindow(run_id="pair_S3c", source_side=row["source_side"], start_s=float(start), duration_s=float(dur), end_s=float(end))
    gt = ground_truth_trajectory(df_parquet, window)
    
    # IMU cache
    cache_df = pd.read_csv(f"data/processed/_cpp_replay_cache/{oid}.csv", comment="#")
    dt = np.diff(cache_df["timestamp_s"].values)
    
    # Physical accel a_lat (v2)
    # in v2, a_lat = |ax - b_accel_x|
    a_lat_raw = np.abs(cache_df["ax"].values)
    
    # Gyro yaw rate
    w_yaw_raw = np.abs(cache_df["gz"].values)
    
    # GT speed and turn
    gt_t = gt["timestamp_s"].values
    gt_hdg = gt["heading_deg"].values
    gt_spd = gt["speed_ms"].values
    total_gt_turn = np.degrees(np.abs(np.unwrap(np.radians(gt_hdg))[-1] - np.unwrap(np.radians(gt_hdg))[0]))
    
    # GT centripetal accel
    gt_dt = np.diff(gt_t)
    gt_wyaw = np.abs(np.diff(np.unwrap(np.radians(gt_hdg))) / gt_dt)
    gt_ac = gt_spd[:-1] * gt_wyaw
    
    print(f"\n{oid} (Duration {dur:3.0f}s, start={start:.1f}s):")
    print(f"  GT: Total Turn = {total_gt_turn:6.1f}°, Mean Speed = {np.mean(gt_spd):5.1f} m/s, Max Speed = {np.max(gt_spd):5.1f} m/s")
    print(f"  GT a_c:  mean = {np.mean(gt_ac):5.2f} m/s², max = {np.max(gt_ac):5.2f} m/s², 90th pct = {np.percentile(gt_ac, 90):5.2f} m/s²")
    print(f"  IMU gz:  mean = {np.degrees(np.mean(w_yaw_raw)):5.1f}°/s, max = {np.degrees(np.max(w_yaw_raw)):5.1f}°/s")
    print(f"  IMU ax (v2 signal): mean = {np.mean(a_lat_raw):5.2f} m/s², max = {np.max(a_lat_raw):5.2f} m/s², 90th pct = {np.percentile(a_lat_raw, 90):5.2f} m/s²")
