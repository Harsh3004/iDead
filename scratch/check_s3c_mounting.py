import pandas as pd
import numpy as np
from dataeval.estimation.ekf import quat_to_rot_matrix

att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#")
att = att_df[att_df["run_id"] == "pair_S3c"].iloc[0]

q0 = np.array([att["q_w"], att["q_x"], att["q_y"], att["q_z"]])
R = quat_to_rot_matrix(q0)

print("Initial Attitude for pair_S3c:")
print(f"  q0 = [{q0[0]:.4f}, {q0[1]:.4f}, {q0[2]:.4f}, {q0[3]:.4f}]")
print(f"  yaw_deg = {att['yaw_deg']:.2f}°, pitch_deg = {att['pitch_deg']:.2f}°, roll_deg = {att['roll_deg']:.2f}°")
print("\nRotation Matrix R (body to ENU):")
print(R)

# In ENU:
# Axis 0 = East, Axis 1 = North, Axis 2 = Up
# R columns are the body axes in ENU:
# Col 0 = body X in ENU
# Col 1 = body Y in ENU
# Col 2 = body Z in ENU
print("\nBody Axes in ENU:")
print(f"  Body X in ENU: [{R[0,0]:+.3f}, {R[1,0]:+.3f}, {R[2,0]:+.3f}]")
print(f"  Body Y in ENU: [{R[0,1]:+.3f}, {R[1,1]:+.3f}, {R[2,1]:+.3f}]")
print(f"  Body Z in ENU: [{R[0,2]:+.3f}, {R[1,2]:+.3f}, {R[2,2]:+.3f}]")

# Gravity vector in body: R^T @ [0, 0, -9.81]
g_b = R.T @ np.array([0, 0, -9.81])
print(f"\nExpected Gravity in Body frame (R^T @ [0, 0, -g]):")
print(f"  g_b = [{g_b[0]:+.2f}, {g_b[1]:+.2f}, {g_b[2]:+.2f}]")

# Check cache initial accel
cache_df = pd.read_csv("data/processed/_cpp_replay_cache/pair_S3c__120s__0.csv", comment="#")
print(f"\nMeasured mean accel in cache:")
print(f"  ax = {cache_df['ax'].mean():+.2f}, ay = {cache_df['ay'].mean():+.2f}, az = {cache_df['az'].mean():+.2f}")
