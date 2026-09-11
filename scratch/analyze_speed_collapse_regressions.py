import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

reg = pd.read_csv("scratch/step23_regressed_manifest.csv")
manifest = pd.read_csv("data/processed/outages/manifest.csv")

print(f"Analyzing speed estimation across {len(reg)} regressed instances (v3 vs v2)...")

results = []
for _, row in reg.iterrows():
    oid = row["outage_id"]
    m_row = manifest[manifest["outage_id"] == oid].iloc[0]
    split = m_row["split"]
    
    pred_path = Path(f"data/processed/cpp_predictions_ekf_v3/{oid}.csv")
    if not pred_path.exists():
        continue
    pred = load_cpp_prediction(pred_path)
    
    parquet_path = Path(f"data/processed/outages/{split}/{oid}.parquet")
    df_parquet = pd.read_parquet(parquet_path)
    window = OutageWindow(
        run_id=m_row["run_or_pair_id"], source_side=m_row["source_side"],
        start_s=float(m_row["start_s"]), duration_s=float(m_row["outage_s"]), end_s=float(m_row["end_s"])
    )
    gt = ground_truth_trajectory(df_parquet, window)
    if gt is None or len(gt) < 2:
        continue
        
    gt_speed = np.interp(pred["timestamp_s"], gt["timestamp_s"], gt["speed_ms"])
    pred_speed = pred["speed_ms"].values
    
    mean_gt = float(np.mean(gt_speed))
    mean_pred = float(np.mean(pred_speed))
    min_pred = float(np.min(pred_speed))
    
    # Speed ratio
    ratio = mean_pred / mean_gt if mean_gt > 0.5 else 1.0
    
    results.append({
        "outage_id": oid,
        "run_id": row["run_id"],
        "outage_s": row["outage_s"],
        "diff_v3_v2": row["diff_v3_v2"],
        "mean_gt_speed": mean_gt,
        "mean_pred_speed": mean_pred,
        "speed_ratio": ratio,
        "collapsed": ratio < 0.5,
        "severely_collapsed": ratio < 0.25,
    })

res_df = pd.DataFrame(results)
print(f"Successfully processed {len(res_df)} instances.")

print("\n--- Speed Collapse Summary on Regressed Instances ---")
print(f"Total instances with Mean Speed Ratio < 0.50: {len(res_df[res_df['collapsed']])} / {len(res_df)} ({len(res_df[res_df['collapsed']])/len(res_df)*100:.1f}%)")
print(f"Total instances with Mean Speed Ratio < 0.25: {len(res_df[res_df['severely_collapsed']])} / {len(res_df)} ({len(res_df[res_df['severely_collapsed']])/len(res_df)*100:.1f}%)")

print("\n--- Breakdown by Duration ---")
for d in [10, 30, 60, 120, 180]:
    sub = res_df[res_df["outage_s"] == d]
    coll = len(sub[sub["collapsed"]])
    sev = len(sub[sub["severely_collapsed"]])
    print(f"Duration {d:3d}s (N={len(sub):2d}): collapsed={coll} ({coll/len(sub)*100:.1f}%), severely_collapsed={sev} ({sev/len(sub)*100:.1f}%), mean ratio={sub['speed_ratio'].mean():.2f}")

