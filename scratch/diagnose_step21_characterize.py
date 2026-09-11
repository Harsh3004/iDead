"""Characterize v1-vs-v2 regression set across all 253 instances and 5 durations."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load leaderboard
df = pd.read_csv("results/leaderboard.csv")
v1 = df[df["config"] == "ekf_zupt_nhc_v1"].set_index("outage_id")
v2 = df[df["config"] == "ekf_zupt_nhc_v2"].set_index("outage_id")
manifest = pd.read_csv("data/processed/outages/manifest.csv").set_index("outage_id")

common_ids = v1.index.intersection(v2.index)
print(f"Total paired instances evaluated: {len(common_ids)}")

records = []
for oid in common_ids:
    r1 = v1.loc[oid]
    r2 = v2.loc[oid]
    man = manifest.loc[oid] if oid in manifest.index else None
    
    err1 = r1["final_pos_error_m"]
    err2 = r2["final_pos_error_m"]
    diff = err2 - err1 # positive means v2 is worse (regressed)
    
    records.append({
        "outage_id": oid,
        "run_id": r1["run_id"],
        "outage_s": r1["outage_s"],
        "split": r1["split"],
        "start_s": man["start_s"] if man is not None else np.nan,
        "end_s": man["end_s"] if man is not None else np.nan,
        "err_v1": err1,
        "err_v2": err2,
        "diff": diff,
        "pct_change": ((err2 - err1) / err1) * 100.0 if err1 > 0 else 0.0,
        "status": "improved" if diff < -0.1 else ("regressed" if diff > 0.1 else "unchanged")
    })

res = pd.DataFrame(records)

print("\n--- Overall Summary ---")
print(res["status"].value_counts())
print(res["status"].value_counts(normalize=True) * 100.0)

print("\n--- By Duration ---")
for d, group in res.groupby("outage_s"):
    n = len(group)
    imp = (group["status"] == "improved").sum()
    reg = (group["status"] == "regressed").sum()
    unc = (group["status"] == "unchanged").sum()
    med_diff = group["diff"].median()
    mean_diff = group["diff"].mean()
    med_gain = (-group[group["status"] == "improved"]["diff"]).median()
    med_loss = group[group["status"] == "regressed"]["diff"].median()
    print(f"Duration {d:3d}s (N={n:2d}): Imp={imp:2d} ({imp/n*100:4.1f}%), Reg={reg:2d} ({reg/n*100:4.1f}%), Unc={unc:2d} ({unc/n*100:4.1f}%) | MedDiff={med_diff:+7.2f}m | MeanDiff={mean_diff:+7.2f}m | MedGain={med_gain:6.2f}m | MedLoss={med_loss:6.2f}m")

print("\n--- Regressed Set Analysis ---")
reg_df = res[res["status"] == "regressed"].copy()
print(f"Total regressed instances: {len(reg_df)}")
print(f"Diff distribution on regressed: min={reg_df['diff'].min():.2f}m, p25={reg_df['diff'].quantile(0.25):.2f}m, p50={reg_df['diff'].median():.2f}m, p75={reg_df['diff'].quantile(0.75):.2f}m, p95={reg_df['diff'].quantile(0.95):.2f}m, max={reg_df['diff'].max():.2f}m")

print("\nTop 15 worst regressed instances:")
worst15 = reg_df.sort_values("diff", ascending=False).head(15)
for _, r in worst15.iterrows():
    print(f"{r['outage_id']:30s} | dur={r['outage_s']:3d}s | start={r['start_s']:6.1f}s | v1={r['err_v1']:8.2f}m -> v2={r['err_v2']:8.2f}m | diff={r['diff']:+8.2f}m ({r['pct_change']:+6.1f}%)")

print("\nTop 15 best improved instances:")
best15 = res[res["status"] == "improved"].sort_values("diff", ascending=True).head(15)
for _, r in best15.iterrows():
    print(f"{r['outage_id']:30s} | dur={r['outage_s']:3d}s | start={r['start_s']:6.1f}s | v1={r['err_v1']:8.2f}m -> v2={r['err_v2']:8.2f}m | diff={r['diff']:+8.2f}m ({r['pct_change']:+6.1f}%)")

print("\n--- Regression by Run ID ---")
run_grp = res.groupby("run_id")["status"].value_counts().unstack(fill_value=0)
if "regressed" in run_grp.columns:
    run_grp["reg_rate"] = run_grp["regressed"] / run_grp.sum(axis=1)
    print("Runs with most regressions:")
    print(run_grp.sort_values(by=["regressed", "reg_rate"], ascending=False).head(20))
