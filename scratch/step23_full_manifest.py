import pandas as pd
import numpy as np

lb = pd.read_csv("results/leaderboard.csv")

# Filter to paired ground truth
from pathlib import Path
from dataeval.harness.run_corrected_replay import load_module_b_tiers
tiers = load_module_b_tiers(Path("results/module_b_initial_attitude.csv"))
eligible = lb[(lb["run_id"].isin(tiers.keys())) & (lb["has_ground_truth"] == True)].copy()

# Pivot on outage_id and config
piv_pos = eligible.pivot(index=["outage_id", "outage_s", "run_id"], columns="config", values="final_pos_error_m").reset_index()
piv_cross = eligible.pivot(index=["outage_id", "outage_s", "run_id"], columns="config", values="cross_track_m").reset_index()
piv_along = eligible.pivot(index=["outage_id", "outage_s", "run_id"], columns="config", values="along_track_m").reset_index()
piv_head = eligible.pivot(index=["outage_id", "outage_s", "run_id"], columns="config", values="heading_error_deg").reset_index()

df = piv_pos.copy()
df["v1"] = df["ekf_zupt_nhc_v1"]
df["v2"] = df["ekf_zupt_nhc_v2"]
df["v3"] = df["ekf_zupt_nhc_v3"]
df["diff_v3_v2"] = df["v3"] - df["v2"]
df["diff_v3_v1"] = df["v3"] - df["v1"]

df["cross_v2"] = piv_cross["ekf_zupt_nhc_v2"]
df["cross_v3"] = piv_cross["ekf_zupt_nhc_v3"]
df["cross_diff"] = df["cross_v3"] - df["cross_v2"]

df["along_v2"] = piv_along["ekf_zupt_nhc_v2"]
df["along_v3"] = piv_along["ekf_zupt_nhc_v3"]
df["along_diff"] = df["along_v3"] - df["along_v2"]

df["head_v2"] = piv_head["ekf_zupt_nhc_v2"]
df["head_v3"] = piv_head["ekf_zupt_nhc_v3"]
df["head_diff"] = df["head_v3"] - df["head_v2"]

print("=== TOTAL PAIRS ===")
print("Total rows:", len(df))

reg = df[df["diff_v3_v2"] > 0.1].copy()
imp = df[df["diff_v3_v2"] < -0.1].copy()
unch = df[df["diff_v3_v2"].abs() <= 0.1].copy()

print(f"Total Improved: {len(imp)} ({len(imp)/len(df)*100:.1f}%)")
print(f"Total Regressed: {len(reg)} ({len(reg)/len(df)*100:.1f}%)")
print(f"Total Unchanged: {len(unch)} ({len(unch)/len(df)*100:.1f}%)")

print("\n=== PER DURATION BREAKDOWN ===")
durations = [10, 30, 60, 120, 180]
for d in durations:
    sub = df[df["outage_s"] == d]
    sub_reg = sub[sub["diff_v3_v2"] > 0.1]
    sub_imp = sub[sub["diff_v3_v2"] < -0.1]
    sub_unch = sub[sub["diff_v3_v2"].abs() <= 0.1]
    
    q = sub_reg["diff_v3_v2"].quantile([0.0, 0.25, 0.50, 0.75, 0.90, 1.0])
    print(f"\nDuration {d}s (N={len(sub)}):")
    print(f"  Improved: {len(sub_imp)}, Regressed: {len(sub_reg)}, Unchanged: {len(sub_unch)}")
    print(f"  Regressed diffs: min={q[0.0]:.2f}m, p25={q[0.25]:.2f}m, med={q[0.50]:.2f}m, p75={q[0.75]:.2f}m, p90={q[0.90]:.2f}m, max={q[1.0]:.2f}m")
    
    # Check along vs cross track changes in regressed
    med_along = (sub_reg["along_v3"].abs() - sub_reg["along_v2"].abs()).median()
    med_cross = (sub_reg["cross_v3"].abs() - sub_reg["cross_v2"].abs()).median()
    med_head = (sub_reg["head_v3"].abs() - sub_reg["head_v2"].abs()).median()
    print(f"  Med change on regressed: |along| {med_along:+.2f}m, |cross| {med_cross:+.2f}m, |heading| {med_head:+.2f}deg")

print("\n=== TOP REGRESSED RUNS (COUNT OF REGRESSIONS ACROSS DURATIONS) ===")
run_counts = reg["run_id"].value_counts()
print(run_counts.head(15))

print("\n=== RUNS REGRESSING AT 3 OR MORE DURATIONS ===")
multi_runs = run_counts[run_counts >= 3].index.tolist()
for r in multi_runs:
    sub_r = df[df["run_id"] == r].sort_values("outage_s")
    print(f"\nRun {r}:")
    for _, row in sub_r.iterrows():
        print(f"  {row['outage_s']:3d}s: v1={row['v1']:7.1f}m, v2={row['v2']:7.1f}m, v3={row['v3']:7.1f}m -> diff(v3-v2)={row['diff_v3_v2']:+7.1f}m, diff(v3-v1)={row['diff_v3_v1']:+7.1f}m | cross_v2={row['cross_v2']:6.1f}, cross_v3={row['cross_v3']:6.1f}, head_v2={row['head_v2']:5.1f}°, head_v3={row['head_v3']:5.1f}°")

reg.to_csv("scratch/step23_regressed_manifest.csv", index=False)
print("\nSaved full regressed manifest to scratch/step23_regressed_manifest.csv")
