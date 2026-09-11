import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

reg = pd.read_csv("scratch/step23_regressed_manifest.csv")
reg_10 = reg[reg["outage_s"] == 10].copy()
manifest = pd.read_csv("data/processed/outages/manifest.csv")

results = []
for _, row in reg_10.iterrows():
    oid = row["outage_id"]
    m_row = manifest[manifest["outage_id"] == oid].iloc[0]
    split = m_row["split"]
    
    p2 = load_cpp_prediction(Path(f"data/processed/cpp_predictions_ekf_v2/{oid}.csv"))
    p3 = load_cpp_prediction(Path(f"data/processed/cpp_predictions_ekf_v3/{oid}.csv"))
    
    # Ground truth
    parquet_path = Path(f"data/processed/outages/{split}/{oid}.parquet")
    df_parquet = pd.read_parquet(parquet_path)
    window = OutageWindow(
        run_id=m_row["run_or_pair_id"], source_side=m_row["source_side"],
        start_s=float(m_row["start_s"]), duration_s=float(m_row["outage_s"]), end_s=float(m_row["end_s"])
    )
    gt = ground_truth_trajectory(df_parquet, window)
    if gt is None or len(gt) < 2:
        continue
        
    gt_speed = gt["speed_ms"].values
    v2_speed = p2["speed_ms"].values
    v3_speed = p3["speed_ms"].values
    
    # Check speed at t=10s
    results.append({
        "oid": oid,
        "run_id": row["run_id"],
        "diff_v3_v2": row["diff_v3_v2"],
        "v2_err": row["v2"],
        "v3_err": row["v3"],
        "gt_spd_end": gt_speed[-1],
        "v2_spd_end": v2_speed[-1],
        "v3_spd_end": v3_speed[-1],
        "gt_spd_mean": np.mean(gt_speed),
        "v2_spd_mean": np.mean(v2_speed),
        "v3_spd_mean": np.mean(v3_speed),
    })

res_df = pd.DataFrame(results)

print(f"--- 10s Regressions Deep Dive (N={len(res_df)}) ---")
print(f"Mean GT Speed:       {res_df['gt_spd_mean'].mean():.2f} m/s")
print(f"Mean v2 Speed:       {res_df['v2_spd_mean'].mean():.2f} m/s (ratio: {res_df['v2_spd_mean'].mean()/res_df['gt_spd_mean'].mean():.2f})")
print(f"Mean v3 Speed:       {res_df['v3_spd_mean'].mean():.2f} m/s (ratio: {res_df['v3_spd_mean'].mean()/res_df['gt_spd_mean'].mean():.2f})")
print(f"\nFinal Speed at t=10s:")
print(f"Mean GT Speed (t=10): {res_df['gt_spd_end'].mean():.2f} m/s")
print(f"Mean v2 Speed (t=10): {res_df['v2_spd_end'].mean():.2f} m/s")
print(f"Mean v3 Speed (t=10): {res_df['v3_spd_end'].mean():.2f} m/s")

# Check under-speeding vs over-speeding
v3_braking = res_df[res_df["v3_spd_end"] < res_df["v2_spd_end"]]
print(f"\nInstances where v3 braked harder than v2: {len(v3_braking)} / {len(res_df)} ({len(v3_braking)/len(res_df)*100:.1f}%)")

# Check along track error
reg_10_along = reg_10["along_diff"]
print(f"\nAlong track diff (v3 - v2) distribution at 10s:")
print(f"  Negative (v3 traveled less distance / braked more): {len(reg_10[reg_10['along_diff'] < 0])} / {len(reg_10)} ({len(reg_10[reg_10['along_diff'] < 0])/len(reg_10)*100:.1f}%)")
