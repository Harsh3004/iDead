"""Comprehensive statistical breakdown of v2-worse-than-v1 regression set."""

import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv("results/leaderboard.csv")
v1 = df[df["config"] == "ekf_zupt_nhc_v1"].set_index("outage_id")
v2 = df[df["config"] == "ekf_zupt_nhc_v2"].set_index("outage_id")
manifest = pd.read_csv("data/processed/outages/manifest.csv").set_index("outage_id")

common_ids = v1.index.intersection(v2.index)

records = []
for oid in common_ids:
    r1 = v1.loc[oid]
    r2 = v2.loc[oid]
    man = manifest.loc[oid]
    
    # Load cache IMU to compute speed and dynamics
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{oid}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    import re
    m = dict(re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line1))
    v0 = float(m.get("speed_ms", 0.0))
    
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    mean_ax = np.mean(np.abs(df_imu["ax"]))
    max_ax = np.max(np.abs(df_imu["ax"]))
    mean_gz = np.mean(np.abs(df_imu["gz"]))
    max_gz = np.max(np.abs(df_imu["gz"]))
    
    err1 = r1["final_pos_error_m"]
    err2 = r2["final_pos_error_m"]
    diff = err2 - err1
    
    # Driving context by run prefix or speed
    run_id = r1["run_id"]
    if run_id.startswith("pair_S") or run_id.startswith("pair_M"):
        env = "Highway / Motorway"
    elif run_id.startswith("pair_Vta") or run_id.startswith("pair_Vtb"):
        env = "Urban / Suburban"
    elif run_id.startswith("pair_Vw") or run_id.startswith("pair_Y"):
        env = "Mixed Road / Rural"
    else:
        env = "Other"
        
    speed_cat = "Low (<5 m/s)" if v0 < 5.0 else ("Med (5-15 m/s)" if v0 <= 15.0 else "High (>15 m/s)")
    timing_cat = "Early (<120s)" if float(man["start_s"]) < 120.0 else "Late (>=120s)"
    
    status = "improved" if diff < -0.1 else ("regressed" if diff > 0.1 else "unchanged")
    
    records.append({
        "outage_id": oid,
        "run_id": run_id,
        "outage_s": int(r1["outage_s"]),
        "env": env,
        "speed_cat": speed_cat,
        "timing_cat": timing_cat,
        "start_s": float(man["start_s"]),
        "v0": v0,
        "mean_ax": mean_ax,
        "max_ax": max_ax,
        "mean_gz": mean_gz,
        "max_gz": max_gz,
        "err_v1": err1,
        "err_v2": err2,
        "diff": diff,
        "status": status,
    })

res = pd.DataFrame(records)

print("=== 1. SUMMARY BY DRIVING ENVIRONMENT ===")
for env, grp in res.groupby("env"):
    n = len(grp)
    imp = (grp["status"] == "improved").sum()
    reg = (grp["status"] == "regressed").sum()
    unc = (grp["status"] == "unchanged").sum()
    print(f"{env:20s} (N={n:3d}): Imp={imp:2d} ({imp/n*100:5.1f}%), Reg={reg:2d} ({reg/n*100:5.1f}%), Unc={unc:2d} ({unc/n*100:5.1f}%) | MedDiff={grp['diff'].median():+6.2f}m")

print("\n=== 2. SUMMARY BY INITIAL SPEED CATEGORY ===")
for spd, grp in res.groupby("speed_cat"):
    n = len(grp)
    imp = (grp["status"] == "improved").sum()
    reg = (grp["status"] == "regressed").sum()
    unc = (grp["status"] == "unchanged").sum()
    print(f"{spd:20s} (N={n:3d}): Imp={imp:2d} ({imp/n*100:5.1f}%), Reg={reg:2d} ({reg/n*100:5.1f}%), Unc={unc:2d} ({unc/n*100:5.1f}%) | MedDiff={grp['diff'].median():+6.2f}m")

print("\n=== 3. SUMMARY BY OUTAGE TIMING (EARLY VS LATE IN RUN) ===")
for tim, grp in res.groupby("timing_cat"):
    n = len(grp)
    imp = (grp["status"] == "improved").sum()
    reg = (grp["status"] == "regressed").sum()
    unc = (grp["status"] == "unchanged").sum()
    print(f"{tim:20s} (N={n:3d}): Imp={imp:2d} ({imp/n*100:5.1f}%), Reg={reg:2d} ({reg/n*100:5.1f}%), Unc={unc:2d} ({unc/n*100:5.1f}%) | MedDiff={grp['diff'].median():+6.2f}m")

print("\n=== 4. CROSS-TABULATION: DURATION x ENVIRONMENT (REGRESSION COUNTS / TOTAL) ===")
ct = pd.crosstab(res["outage_s"], res["env"], values=res["status"].apply(lambda s: 1 if s == "regressed" else 0), aggfunc=["sum", "count"])
print(ct)

print("\n=== 5. LATERAL ACCELERATION NOISE COMPARISON: IMPROVED VS REGRESSED ===")
imp_grp = res[res["status"] == "improved"]
reg_grp = res[res["status"] == "regressed"]
print(f"Improved  (N={len(imp_grp)}): Mean |ax| = {imp_grp['mean_ax'].mean():.3f} m/s^2, Mean Max |ax| = {imp_grp['max_ax'].mean():.3f} m/s^2, Mean |gz| = {imp_grp['mean_gz'].mean():.3f} rad/s")
print(f"Regressed (N={len(reg_grp)}): Mean |ax| = {reg_grp['mean_ax'].mean():.3f} m/s^2, Mean Max |ax| = {reg_grp['max_ax'].mean():.3f} m/s^2, Mean |gz| = {reg_grp['mean_gz'].mean():.3f} rad/s")

# Let's save res for further analysis
res.to_csv("scratch/step21_breakdown_full.csv", index=False)
