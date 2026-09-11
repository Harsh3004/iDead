import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

reg = pd.read_csv("scratch/step23_regressed_manifest.csv")
manifest = pd.read_csv("data/processed/outages/manifest.csv")

# Filter to 60s and 120s regressions
reg_60_120 = reg[reg["outage_s"].isin([60, 120])].copy()
print(f"Analyzing {len(reg_60_120)} regressed instances at 60s & 120s (28 at 60s, 22 at 120s)...")

results = []
for _, row in reg_60_120.iterrows():
    oid = row["outage_id"]
    m_row = manifest[manifest["outage_id"] == oid].iloc[0]
    split = m_row["split"]
    
    parquet_path = Path(f"data/processed/outages/{split}/{oid}.parquet")
    df_parquet = pd.read_parquet(parquet_path)
    window = OutageWindow(
        run_id=m_row["run_or_pair_id"], source_side=m_row["source_side"],
        start_s=float(m_row["start_s"]), duration_s=float(m_row["outage_s"]), end_s=float(m_row["end_s"])
    )
    gt = ground_truth_trajectory(df_parquet, window)
    if gt is None or len(gt) < 2:
        continue
        
    t = gt["timestamp_s"].values
    hdg = gt["heading_deg"].values
    speed = gt["speed_ms"].values
    
    dt = np.diff(t)
    w_yaw = np.abs(np.diff(np.unwrap(np.radians(hdg))) / dt)
    a_c_gt = speed[:-1] * w_yaw
    
    total_turn_deg = float(np.abs(np.degrees(np.unwrap(np.radians(hdg))[-1] - np.unwrap(np.radians(hdg))[0])))
    mean_ac = float(np.mean(a_c_gt))
    max_ac = float(np.max(a_c_gt))
    
    # Read v2 and v3 predictions
    pred_v2 = load_cpp_prediction(Path(f"data/processed/cpp_predictions_ekf_v2/{oid}.csv"))
    pred_v3 = load_cpp_prediction(Path(f"data/processed/cpp_predictions_ekf_v3/{oid}.csv"))
    
    v2_spd_mean = float(pred_v2["speed_ms"].mean())
    v3_spd_mean = float(pred_v3["speed_ms"].mean())
    gt_spd_mean = float(np.mean(speed))
    
    # Classify trajectory
    # If total turn > 30 deg or max_ac > 1.0 m/s2: Curved Road
    is_curved = (total_turn_deg > 30.0) or (max_ac > 1.0)
    
    results.append({
        "outage_id": oid,
        "run_id": row["run_id"],
        "outage_s": row["outage_s"],
        "v2_err": row["v2"],
        "v3_err": row["v3"],
        "diff_v3_v2": row["diff_v3_v2"],
        "total_turn_deg": total_turn_deg,
        "mean_ac_gt": mean_ac,
        "max_ac_gt": max_ac,
        "is_curved": is_curved,
        "gt_spd": gt_spd_mean,
        "v2_spd": v2_spd_mean,
        "v3_spd": v3_spd_mean,
    })

res_df = pd.DataFrame(results)

curved_runs = res_df[res_df["is_curved"]]
straight_runs = res_df[~res_df["is_curved"]]

print(f"\n--- Trajectory Classification of 60s & 120s Regressions (N={len(res_df)}) ---")
print(f"Curved Roads (Turn > 30° or a_c > 1.0 m/s²):   {len(curved_runs)} ({len(curved_runs)/len(res_df)*100:.1f}%)")
print(f"Straight/Gentle Roads:                         {len(straight_runs)} ({len(straight_runs)/len(res_df)*100:.1f}%)")

print("\n--- CURVED ROADS (Where genuine turn relaxation is needed) ---")
print(f"Count: {len(curved_runs)}")
print(f"Mean GT Speed: {curved_runs['gt_spd'].mean():.1f} m/s")
print(f"Mean v2 Speed: {curved_runs['v2_spd'].mean():.1f} m/s (ratio: {curved_runs['v2_spd'].mean()/curved_runs['gt_spd'].mean():.2f})")
print(f"Mean v3 Speed: {curved_runs['v3_spd'].mean():.1f} m/s (ratio: {curved_runs['v3_spd'].mean()/curved_runs['gt_spd'].mean():.2f})")
print(f"Median Regression Magnitude: {curved_runs['diff_v3_v2'].median():.1f}m")

print("\n--- STRAIGHT/GENTLE ROADS (Where vibration in v2 was active) ---")
print(f"Count: {len(straight_runs)}")
print(f"Mean GT Speed: {straight_runs['gt_spd'].mean():.1f} m/s")
print(f"Mean v2 Speed: {straight_runs['v2_spd'].mean():.1f} m/s (ratio: {straight_runs['v2_spd'].mean()/straight_runs['gt_spd'].mean():.2f})")
print(f"Mean v3 Speed: {straight_runs['v3_spd'].mean():.1f} m/s (ratio: {straight_runs['v3_spd'].mean()/straight_runs['gt_spd'].mean():.2f})")
print(f"Median Regression Magnitude: {straight_runs['diff_v3_v2'].median():.1f}m")

print("\nSample Straight Regressions where v2 beat v3:")
print(straight_runs[["outage_id", "outage_s", "diff_v3_v2", "total_turn_deg", "v2_spd", "v3_spd", "gt_spd"]].head(10))

