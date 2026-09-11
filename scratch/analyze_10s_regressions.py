import pandas as pd
import numpy as np

reg = pd.read_csv("scratch/step23_regressed_manifest.csv")
reg_10 = reg[reg["outage_s"] == 10].copy()

print(f"Analyzing {len(reg_10)} regressed instances at 10s...")
print("diff_v3_v2 distribution at 10s:")
print(reg_10["diff_v3_v2"].describe())

# Check along vs cross track
print("\nAlong-track diff (v3 - v2) at 10s:")
print(reg_10["along_diff"].describe())

print("\nCross-track diff (v3 - v2) at 10s:")
print(reg_10["cross_diff"].describe())

print("\nHeading diff (v3 - v2) at 10s:")
print(reg_10["head_diff"].describe())

# Compare v1 vs v2 vs v3 on these 37 instances
print("\nMean errors on 10s regressed:")
print(f"  v1 err: {reg_10['v1'].mean():.2f}m")
print(f"  v2 err: {reg_10['v2'].mean():.2f}m")
print(f"  v3 err: {reg_10['v3'].mean():.2f}m")

# Check if v3 matches v1 on these 10s instances
diff_v3_v1 = reg_10["v3"] - reg_10["v1"]
print(f"\nDiff (v3 - v1) on 10s regressed:")
print(diff_v3_v1.describe())
