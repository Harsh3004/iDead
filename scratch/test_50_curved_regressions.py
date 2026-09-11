import pandas as pd
import numpy as np
from pathlib import Path
from scratch.test_step24_prototype import simulate_v4

reg = pd.read_csv("scratch/step23_regressed_manifest.csv")
reg_60_120 = reg[reg["outage_s"].isin([60, 120])].copy()
print(f"Testing Step 24 Option A (nav_z + speed floor + 2.0Hz LPF) on {len(reg_60_120)} curved regressions...")

results = []
for _, r in reg_60_120.iterrows():
    oid = r["outage_id"]
    err_v2 = r["v2"]
    err_v3 = r["v3"]
    err_v4 = simulate_v4(oid, kc=2.0, cutoff_hz=2.0, use_speed_floor=True, use_nav_yaw=True)
    results.append({
        "oid": oid,
        "outage_s": r["outage_s"],
        "run_id": r["run_id"],
        "v2": err_v2,
        "v3": err_v3,
        "v4": err_v4,
        "diff_v4_v3": err_v4 - err_v3,
        "diff_v4_v2": err_v4 - err_v2,
    })

res_df = pd.DataFrame(results)
imp_vs_v3 = res_df[res_df["diff_v4_v3"] < -0.1]
beat_v2 = res_df[res_df["diff_v4_v2"] < -0.1]

print("\n--- Summary on 60s & 120s Regressions (N=50) ---")
print(f"Improved vs v3: {len(imp_vs_v3)} / {len(res_df)} ({len(imp_vs_v3)/len(res_df)*100:.1f}%)")
print(f"Beat v2:        {len(beat_v2)} / {len(res_df)} ({len(beat_v2)/len(res_df)*100:.1f}%)")
print(f"Median change vs v3: {res_df['diff_v4_v3'].median():.2f}m")
print(f"Median change vs v2: {res_df['diff_v4_v2'].median():.2f}m")
