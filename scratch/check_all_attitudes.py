import pandas as pd
import numpy as np

df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#")
print(f"Total runs in attitude CSV: {len(df)}")
print("Pitch (deg) summary:")
print(df["pitch_deg"].describe())
print("\nRoll (deg) summary:")
print(df["roll_deg"].describe())

# Check how many runs have pitch or roll > 30 deg
tilted = df[(df["pitch_deg"].abs() > 30) | (df["roll_deg"].abs() > 30)]
print(f"\nRuns with |pitch| > 30° or |roll| > 30°: {len(tilted)} / {len(df)}")
for _, r in tilted.iterrows():
    print(f"  {r['run_id']:12s}: pitch={r['pitch_deg']:6.1f}°, roll={r['roll_deg']:6.1f}°, yaw={r['yaw_deg']:6.1f}°")
