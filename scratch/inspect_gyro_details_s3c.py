import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/_cpp_replay_cache/pair_S3c__120s__0.csv", comment="#")
t = df["timestamp_s"].values
t_rel = t - t[0]

print("Timestamps: start=", t[0], "end=", t[-1], "duration=", t[-1]-t[0])
print("\nSample every 15s:")
print(f"{'t(s)':5s} | {'ax':6s} | {'ay':6s} | {'az':6s} | {'gx (deg/s)':10s} | {'gy (deg/s)':10s} | {'gz (deg/s)':10s}")
print("-" * 65)
for step in range(0, 121, 15):
    idx = np.argmin(np.abs(t_rel - step))
    print(f"{step:4d}s | {df['ax'].iloc[idx]:+6.2f} | {df['ay'].iloc[idx]:+6.2f} | {df['az'].iloc[idx]:+6.2f} | {np.degrees(df['gx'].iloc[idx]):+10.2f} | {np.degrees(df['gy'].iloc[idx]):+10.2f} | {np.degrees(df['gz'].iloc[idx]):+10.2f}")

# Also check module B initial attitude
att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#")
print("\nModule B initial attitude for pair_S3c:")
print(att_df[att_df["run_id"] == "pair_S3c"][["run_id", "yaw_deg", "pitch_deg", "roll_deg", "q_w", "q_x", "q_y", "q_z", "gyro_bias_x_rad_s", "gyro_bias_y_rad_s", "gyro_bias_z_rad_s"]])
