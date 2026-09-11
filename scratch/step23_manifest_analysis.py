import pandas as pd
import numpy as np

reg = pd.read_csv("scratch/step23_regressed_manifest.csv")
lb = pd.read_csv("results/leaderboard.csv")
manifest = pd.read_csv("data/processed/outages/manifest.csv")

print("================================================================================")
print("SECTION 1: REGRESSION MANIFEST & DISTRIBUTION ANALYSIS")
print("================================================================================")
print(f"Total Evaluated Instances: 253")
print(f"Total Regressed (v3 > v2 + 0.1m): {len(reg)} ({len(reg)/253*100:.1f}%)")

# Breakdown by duration
durations = [10, 30, 60, 120, 180]
print("\n--- Distribution of Regression Magnitude (diff = v3 - v2 [m]) by Duration ---")
print(f"{'Duration':8s} | {'N_eval':6s} | {'N_reg':5s} | {'% Reg':6s} | {'Min':7s} | {'p25':7s} | {'Median':7s} | {'p75':7s} | {'p90':7s} | {'Max':7s}")
print("-" * 80)
for d in durations:
    sub_eval = lb[(lb["config"] == "ekf_zupt_nhc_v3") & (lb["outage_s"] == d) & (lb["has_ground_truth"] == True)]
    sub = reg[reg["outage_s"] == d]
    q = sub["diff_v3_v2"].quantile([0.0, 0.25, 0.50, 0.75, 0.90, 1.0])
    pct = len(sub) / len(sub_eval) * 100
    print(f"{d:3d}s     | {len(sub_eval):6d} | {len(sub):5d} | {pct:5.1f}% | {q[0.0]:6.2f}m | {q[0.25]:6.2f}m | {q[0.50]:6.2f}m | {q[0.75]:6.2f}m | {q[0.90]:6.2f}m | {q[1.0]:6.2f}m")

# Error components on regressed instances
print("\n--- Error Breakdown on Regressed Instances (Median v3 vs v2) ---")
print(f"{'Duration':8s} | {'Med v2 Err':10s} | {'Med v3 Err':10s} | {'Med |along| Diff':16s} | {'Med |cross| Diff':16s} | {'Med |head| Diff':15s}")
print("-" * 80)
for d in durations:
    sub = reg[reg["outage_s"] == d]
    med_v2 = sub["v2"].median()
    med_v3 = sub["v3"].median()
    d_along = (sub["along_v3"].abs() - sub["along_v2"].abs()).median()
    d_cross = (sub["cross_v3"].abs() - sub["cross_v2"].abs()).median()
    d_head = (sub["head_v3"].abs() - sub["head_v2"].abs()).median()
    print(f"{d:3d}s     | {med_v2:9.1f}m | {med_v3:9.1f}m | {d_along:+15.2f}m | {d_cross:+15.2f}m | {d_head:+14.2f}°")

# Run Concentration
print("\n--- Run ID Concentration ---")
rc = reg["run_id"].value_counts()
print(f"Unique runs evaluated: 72")
print(f"Unique runs with at least 1 regression: {len(rc)}")
print("Distribution of regressions per run:")
print(f"  4 regressions: {len(rc[rc == 4])} runs ({', '.join(rc[rc == 4].index)})")
print(f"  3 regressions: {len(rc[rc == 3])} runs")
print(f"  2 regressions: {len(rc[rc == 2])} runs")
print(f"  1 regression:  {len(rc[rc == 1])} runs")
