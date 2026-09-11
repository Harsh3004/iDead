import pandas as pd

df = pd.read_csv("scratch/step23_regressed_manifest.csv")
print(f"Total Regressed Instances (v3 vs v2): {len(df)}")

print("\n--- Regression Magnitude by Duration (diff = v3 - v2 [m]) ---")
for d in [10, 30, 60, 120, 180]:
    sub = df[df["outage_s"] == d]
    q = sub["diff_v3_v2"].quantile([0.0, 0.25, 0.50, 0.75, 0.90, 1.0])
    print(f"Duration {d:3d}s (N={len(sub):2d}): min={q[0.0]:6.2f}m, p25={q[0.25]:6.2f}m, med={q[0.50]:6.2f}m, p75={q[0.75]:6.2f}m, p90={q[0.90]:6.2f}m, max={q[1.0]:6.2f}m")

print("\n--- Run ID Frequency in Regressions ---")
rc = df["run_id"].value_counts()
print(f"Unique runs with regressions: {len(rc)}")
print(f"Runs with 4-5 regressions: {len(rc[rc >= 4])}")
print(f"Runs with 3 regressions:   {len(rc[rc == 3])}")
print(f"Runs with 2 regressions:   {len(rc[rc == 2])}")
print(f"Runs with 1 regression:    {len(rc[rc == 1])}")

print("\nTop 10 Runs by Number of Regressions:")
for r, c in rc.head(10).items():
    print(f"  {r}: {c} regressions")
