import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/_cpp_replay_cache/pair_S1__180s__0.csv", comment="#")
att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#").set_index("run_id")
att = att_df.loc["pair_S1"]
gb0 = np.array([att["gyro_bias_x_rad_s"], att["gyro_bias_y_rad_s"], att["gyro_bias_z_rad_s"]])

gx = df["gx"].values - gb0[0]
gy = df["gy"].values - gb0[1]
gz = df["gz"].values - gb0[2]

norm_w = np.sqrt(gx**2 + gy**2 + gz**2)

print("pair_S1 Unbiased Gyros (deg/s):")
print(f"  gx:   mean={np.degrees(np.mean(gx)):.2f}°, std={np.degrees(np.std(gx)):.2f}°, max={np.degrees(np.max(np.abs(gx))):.2f}°")
print(f"  gy:   mean={np.degrees(np.mean(gy)):.2f}°, std={np.degrees(np.std(gy)):.2f}°, max={np.degrees(np.max(np.abs(gy))):.2f}°")
print(f"  gz:   mean={np.degrees(np.mean(gz)):.2f}°, std={np.degrees(np.std(gz)):.2f}°, max={np.degrees(np.max(np.abs(gz))):.2f}°")
print(f"  norm: mean={np.degrees(np.mean(norm_w)):.2f}°, std={np.degrees(np.std(norm_w)):.2f}°, max={np.degrees(np.max(norm_w)):.2f}°")

# Filtered norm with 2.0 Hz LPF
dt = 0.1 # 10Hz
tau = 1.0 / (2.0 * np.pi * 2.0)
alpha = dt / (dt + tau)
filt_norm = np.zeros(len(norm_w))
filt_norm[0] = norm_w[0]
for i in range(1, len(norm_w)):
    filt_norm[i] = filt_norm[i-1] + alpha * (norm_w[i] - filt_norm[i-1])

print(f"  filtered norm: mean={np.degrees(np.mean(filt_norm)):.2f}°, max={np.degrees(np.max(filt_norm)):.2f}°")

# Now check: In pair_S1, which axis carries the turn?
# Earlier check_all_runs_gyro_axis.py said pair_S1 has GT turn = 528.8 deg, gy integrated = -542.3 deg!
# WAIT! pair_S1 HAS A 528 DEGREE TURN?!
