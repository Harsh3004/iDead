import pandas as pd
import numpy as np

manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
df_parquet = pd.read_parquet(f"data/processed/outages/{row['split']}/pair_S3c__120s__0.parquet")
window = OutageWindow(
    run_id=row["run_or_pair_id"],
    source_side=row["source_side"],
    start_s=float(row["start_s"]),
    duration_s=float(row["outage_s"]),
    end_s=float(row["end_s"]),
)
gt = ground_truth_trajectory(df_parquet, window)

print("GT start point:", gt["latitude_deg"].iloc[0], gt["longitude_deg"].iloc[0])
print("GT end point:  ", gt["latitude_deg"].iloc[-1], gt["longitude_deg"].iloc[-1])
print("GT start heading:", gt["heading_deg"].iloc[0])
print("GT end heading:  ", gt["heading_deg"].iloc[-1])

# Check heading evolution
hdg = gt["heading_deg"].values
t = gt["timestamp_s"].values
print("\nHeading every 10s:")
for i in range(0, len(t), 100):
    print(f"t={t[i]-t[0]:5.1f}s, hdg={hdg[i]:6.1f}°")
