import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_to_rot_matrix

# Parse initial state from cache file
cache_path = "data/processed/_cpp_replay_cache/pair_S3c__120s__0.csv"
with open(cache_path, "r") as f:
    lines = [f.readline() for _ in range(2)]

# Parse initial state
prefix = "# initial_state:"
init_dict = {}
for tok in lines[0].replace(prefix, "").strip().split(","):
    if "=" in tok:
        k, v = tok.split("=")
        init_dict[k.strip()] = float(v.strip())

# Parse attitude from module_b_initial_attitude.csv
att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#")
att_row = att_df[att_df["run_id"] == "pair_S3c"].iloc[0]

q0 = np.array([att_row["q_w"], att_row["q_x"], att_row["q_y"], att_row["q_z"]], dtype=float)
gb0 = np.array([att_row["gyro_bias_x_rad_s"], att_row["gyro_bias_y_rad_s"], att_row["gyro_bias_z_rad_s"]], dtype=float)
init_accel = np.array([init_dict["ax0"], init_dict["ay0"], init_dict["az0"]], dtype=float)

# Load IMU samples
df_imu = pd.read_csv(cache_path, comment="#")

def run_sim(cfg):
    ekf = ErrorStateEkf(cfg)
    ekf.initialize(
        init_dict["t0"], init_dict["lat0"], init_dict["lon0"], init_dict["alt0"],
        init_dict["speed_ms"], init_dict["heading_deg"],
        q0, gb0, np.zeros(3), init_accel
    )
    
    records = []
    for _, row in df_imu.iterrows():
        t = row["timestamp_s"]
        f_b = np.array([row["ax"], row["ay"], row["az"]])
        w_b = np.array([row["gx"], row["gy"], row["gz"]])
        
        ekf.predict(t, f_b, w_b)
        
        # compute NHC sigma before update
        sig_lat = ekf.compute_nhc_sigma_lat()
        
        # body velocity
        R = quat_to_rot_matrix(ekf.q)
        v_b = R.T @ ekf.v_enu
        spd = float(np.linalg.norm(ekf.v_enu))
        
        a_lat_raw = abs(f_b[0] - ekf.b_accel[0])
        w_yaw = abs(w_b[2] - ekf.b_gyro[2])
        a_c = spd * w_yaw
        
        # update NHC
        ekf.update_nhc()
        
        # heading in degrees
        # ENU heading: yaw = atan2(East, North)
        # Body heading: forward is Y (in ENU)
        y_b_nav = R @ np.array([0.0, 1.0, 0.0])
        hdg = np.degrees(np.arctan2(y_b_nav[0], y_b_nav[1])) % 360.0
        
        lat, lon, alt = ekf.get_position_lat_lon()
        records.append({
            "t": t - init_dict["t0"],
            "speed": spd,
            "v_bx": v_b[0],
            "v_by": v_b[1],
            "sig_lat": sig_lat,
            "a_lat_raw": a_lat_raw,
            "a_c": a_c,
            "w_yaw": w_yaw,
            "heading": hdg,
            "lat": lat,
            "lon": lon,
        })
    return pd.DataFrame(records)

cfg_v1 = EkfConfig(nhc_settle_sigma_extra=0.0, nhc_curv_c_coeff=0.0, nhc_curv_lat_coeff=0.0, nhc_curv_yaw_coeff=0.0)
cfg_v2 = EkfConfig(nhc_settle_duration_s=2.0, nhc_settle_sigma_extra=4.0, nhc_curv_c_coeff=0.0, nhc_curv_lat_coeff=5.0, nhc_curv_yaw_coeff=2.0)
cfg_v3 = EkfConfig(nhc_settle_duration_s=2.0, nhc_settle_sigma_extra=4.0, nhc_curv_c_coeff=2.0, nhc_curv_lat_coeff=0.0, nhc_curv_yaw_coeff=0.0)

res_v1 = run_sim(cfg_v1)
res_v2 = run_sim(cfg_v2)
res_v3 = run_sim(cfg_v3)

print("Simulation finished.")
print("Final positions:")
print(f"  v1: lat={res_v1['lat'].iloc[-1]:.6f}, lon={res_v1['lon'].iloc[-1]:.6f}")
print(f"  v2: lat={res_v2['lat'].iloc[-1]:.6f}, lon={res_v2['lon'].iloc[-1]:.6f}")
print(f"  v3: lat={res_v3['lat'].iloc[-1]:.6f}, lon={res_v3['lon'].iloc[-1]:.6f}")

# Sample comparison every 10s
times = list(range(0, 121, 10))
print("\n--- Sigma NHC Lat Over Time (m/s) ---")
print("  t(s) |    v1 Sig |    v2 Sig |    v3 Sig |  v2 a_lat |   v3 a_c |  w_yaw (rad/s)")
for t in times:
    idx = np.argmin(np.abs(res_v3["t"] - t))
    r1, r2, r3 = res_v1.iloc[idx], res_v2.iloc[idx], res_v3.iloc[idx]
    print(f"  {t:4d} |  {r1['sig_lat']:7.3f} |  {r2['sig_lat']:7.3f} |  {r3['sig_lat']:7.3f} |  {r2['a_lat_raw']:8.3f} | {r3['a_c']:8.3f} | {r3['w_yaw']:8.4f}")

print("\n--- Estimated Speed Over Time (m/s) ---")
print("  t(s) |  v1 Speed |  v2 Speed |  v3 Speed |   v1 v_bx |   v2 v_bx |   v3 v_bx")
for t in times:
    idx = np.argmin(np.abs(res_v3["t"] - t))
    r1, r2, r3 = res_v1.iloc[idx], res_v2.iloc[idx], res_v3.iloc[idx]
    print(f"  {t:4d} |  {r1['speed']:7.2f} |  {r2['speed']:7.2f} |  {r3['speed']:7.2f} |  {r1['v_bx']:8.3f} | {r2['v_bx']:8.3f} | {r3['v_bx']:8.3f}")

