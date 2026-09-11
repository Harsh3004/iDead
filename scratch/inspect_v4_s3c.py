import pandas as pd
import numpy as np
from pathlib import Path
from scratch.test_step24_prototype import EkfV4Prototype, parse_header, manifest, att_df
from dataeval.estimation.ekf import EkfConfig
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

outage_id = "pair_S3c__120s__0"
cache_csv = Path(f"data/processed/_cpp_replay_cache/{outage_id}.csv")
with open(cache_csv, "r") as f:
    line1 = f.readline()
init = parse_header(line1)
df_imu = pd.read_csv(cache_csv, skiprows=1)

run_id = outage_id.split("__")[0]
att = att_df.loc[run_id]

cfg = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0)
ekf = EkfV4Prototype(cfg, kc=2.0, cutoff_hz=2.0)
ekf.v0 = init["speed_ms"]

q0 = np.array([att["q_w"], att["q_x"], att["q_y"], att["q_z"]])
gb0 = np.array([att["gyro_bias_x_rad_s"], att["gyro_bias_y_rad_s"], att["gyro_bias_z_rad_s"]])
init_accel = np.array([init["ax0"], init["ay0"], init["az0"]])

ekf.initialize(
    t0=init["t0"], lat0=init["lat0"], lon0=init["lon0"], alt0=init["alt0"],
    speed_ms=init["speed_ms"], heading_deg=init["heading_deg"],
    q0=q0, b_gyro0=gb0, b_accel0=np.zeros(3), initial_accel=init_accel
)

t_rel_list, e_list, n_list, spd_list, hdg_list, sig_lat_list = [], [], [], [], [], []
lat0, lon0 = np.radians(init["lat0"]), np.radians(init["lon0"])
kEarth = 6371000.0

for _, row in df_imu.iterrows():
    t = row["timestamp_s"]
    f_b = np.array([row["ax"], row["ay"], row["az"]])
    w_b = np.array([row["gx"], row["gy"], row["gz"]])
    ekf.predict(t, f_b, w_b)
    if ekf.is_stationary():
        ekf.update_zupt()
    ekf.update_nhc()
    
    lat, lon, _ = ekf.get_position_lat_lon()
    east = np.radians(lon - lon0) * (kEarth * np.cos(lat0))
    north = np.radians(lat - lat0) * kEarth
    
    t_rel_list.append(t - init["t0"])
    e_list.append(east)
    n_list.append(north)
    spd_list.append(ekf.get_speed_ms())
    hdg_list.append(ekf.get_heading_deg())
    sig_lat_list.append(ekf.compute_nhc_sigma_lat())

man = manifest.loc[outage_id]
df_parquet = pd.read_parquet(f"data/processed/outages/{man['split']}/{outage_id}.parquet")
window = OutageWindow(
    run_id=man["run_or_pair_id"], source_side=man["source_side"],
    start_s=man["start_s"], duration_s=float(man["outage_s"]), end_s=man["end_s"]
)
gt = ground_truth_trajectory(df_parquet, window)
gt_e = np.radians(gt["longitude_deg"] - np.degrees(lon0)) * (kEarth * np.cos(lat0))
gt_n = np.radians(gt["latitude_deg"] - np.degrees(lat0)) * kEarth

print("v4 Trajectory Evolution for pair_S3c__120s__0:")
print(f"{'t(s)':5s} | {'GT (E, N)':20s} | {'v4 (E, N)':20s} | {'GT spd':7s} | {'v4 spd':7s} | {'v4 hdg':8s} | {'sigma_lat':10s} | {'Err':8s}")
print("-" * 105)
for step in range(0, 121, 10):
    idx = np.argmin(np.abs(np.array(t_rel_list) - step))
    gt_idx = np.argmin(np.abs(gt["timestamp_s"].values - (init["t0"] + step)))
    err = np.hypot(e_list[idx] - gt_e.iloc[gt_idx], n_list[idx] - gt_n.iloc[gt_idx])
    print(f"{step:4d}s | ({gt_e.iloc[gt_idx]:+7.1f}, {gt_n.iloc[gt_idx]:+7.1f}) | ({e_list[idx]:+7.1f}, {n_list[idx]:+7.1f}) | {gt['speed_ms'].iloc[gt_idx]:5.1f} | {spd_list[idx]:5.1f} | {hdg_list[idx]:6.1f}° | {sig_lat_list[idx]:8.2f}m/s | {err:7.1f}m")

dist_v4 = np.hypot(e_list[-1] - e_list[0], n_list[-1] - n_list[0])
dist_gt = np.hypot(gt_e.iloc[-1] - gt_e.iloc[0], gt_n.iloc[-1] - gt_n.iloc[0])
print(f"\nFinal net displacement: v4 = {dist_v4:.1f}m, GT = {dist_gt:.1f}m")
