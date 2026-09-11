import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_to_rot_matrix

# Parse initial state from cache file
cache_path = "data/processed/_cpp_replay_cache/pair_S3c__120s__0.csv"
with open(cache_path, "r") as f:
    lines = [f.readline() for _ in range(2)]

prefix = "# initial_state:"
init_dict = {}
for tok in lines[0].replace(prefix, "").strip().split(","):
    if "=" in tok:
        k, v = tok.split("=")
        init_dict[k.strip()] = float(v.strip())

att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#")
att_row = att_df[att_df["run_id"] == "pair_S3c"].iloc[0]

q0 = np.array([att_row["q_w"], att_row["q_x"], att_row["q_y"], att_row["q_z"]], dtype=float)
gb0 = np.array([att_row["gyro_bias_x_rad_s"], att_row["gyro_bias_y_rad_s"], att_row["gyro_bias_z_rad_s"]], dtype=float)
init_accel = np.array([init_dict["ax0"], init_dict["ay0"], init_dict["az0"]], dtype=float)

df_imu = pd.read_csv(cache_path, comment="#")

# Ground truth trajectory
manifest = pd.read_csv("data/processed/outages/manifest.csv")
row = manifest[manifest["outage_id"] == "pair_S3c__120s__0"].iloc[0]
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
df_parquet = pd.read_parquet(f"data/processed/outages/{row['split']}/pair_S3c__120s__0.parquet")
window = OutageWindow(
    run_id=row["run_or_pair_id"], source_side=row["source_side"],
    start_s=float(row["start_s"]), duration_s=float(row["outage_s"]), end_s=float(row["end_s"])
)
gt = ground_truth_trajectory(df_parquet, window)
gt_lat_final = gt["latitude_deg"].iloc[-1]
gt_lon_final = gt["longitude_deg"].iloc[-1]

kEarthRadius = 6371000.0
lat0, lon0 = np.radians(gt["latitude_deg"].iloc[0]), np.radians(gt["longitude_deg"].iloc[0])
gt_east = np.radians(gt_lon_final - lon0) * (kEarthRadius * np.cos(lat0))
gt_north = np.radians(gt_lat_final - lat0) * kEarthRadius

def compute_final_error(lat, lon):
    e = np.radians(lon - lon0) * (kEarthRadius * np.cos(lat0))
    n = np.radians(lat - lat0) * kEarthRadius
    return float(np.hypot(e - gt_east, n - gt_north))

# Interpolate GT speed to IMU timestamps
t_imu = df_imu["timestamp_s"].values
gt_speed_interp = np.interp(t_imu, gt["timestamp_s"], gt["speed_ms"])

def run_custom_ekf(
    kc: float,
    use_gt_speed: bool = False,
    gyro_mode: str = "z", # "z", "norm", or "nav_z"
    klat: float = 0.0,
    kyaw: float = 0.0
):
    cfg = EkfConfig(
        nhc_settle_duration_s=2.0,
        nhc_settle_sigma_extra=4.0,
        nhc_curv_c_coeff=kc,
        nhc_curv_lat_coeff=klat,
        nhc_curv_yaw_coeff=kyaw,
    )
    ekf = ErrorStateEkf(cfg)
    ekf.initialize(
        init_dict["t0"], init_dict["lat0"], init_dict["lon0"], init_dict["alt0"],
        init_dict["speed_ms"], init_dict["heading_deg"],
        q0, gb0, np.zeros(3), init_accel
    )
    
    for i, (_, row) in enumerate(df_imu.iterrows()):
        t = row["timestamp_s"]
        f_b = np.array([row["ax"], row["ay"], row["az"]])
        w_b = np.array([row["gx"], row["gy"], row["gz"]])
        
        ekf.predict(t, f_b, w_b)
        
        # Override compute_nhc_sigma_lat dynamically based on test mode
        dt_init = max(0.0, ekf.t - ekf.t0)
        settle_extra = 0.0
        if ekf.cfg.nhc_settle_duration_s > 1e-6:
            settle_extra = ekf.cfg.nhc_settle_sigma_extra * np.exp(-dt_init / ekf.cfg.nhc_settle_duration_s)
        
        spd = gt_speed_interp[i] if use_gt_speed else float(np.linalg.norm(ekf.v_enu))
        
        w_unbiased = w_b - ekf.b_gyro
        if gyro_mode == "z":
            w_rot = abs(w_unbiased[2])
        elif gyro_mode == "norm":
            w_rot = float(np.linalg.norm(w_unbiased))
        elif gyro_mode == "nav_z":
            R = quat_to_rot_matrix(ekf.q)
            w_nav = R @ w_unbiased
            w_rot = abs(w_nav[2])
        elif gyro_mode == "y":
            w_rot = abs(w_unbiased[1])
        else:
            w_rot = abs(w_unbiased[2])
            
        a_c = spd * w_rot
        c_term = kc * a_c
        yaw_term = kyaw * w_rot
        lat_term = 0.0
        if klat > 1e-6:
            a_lat = abs(f_b[0] - ekf.b_accel[0])
            lat_term = klat * a_lat
            
        curv_scale = np.sqrt(1.0 + c_term * c_term + lat_term * lat_term + yaw_term * yaw_term)
        sigma_lat = (ekf.cfg.sigma_nhc_lat + settle_extra) * curv_scale
        
        # Apply custom NHC update with sigma_lat
        R = quat_to_rot_matrix(ekf.q)
        v_b = R.T @ ekf.v_enu
        z = np.array([-v_b[0], -v_b[2]], dtype=float)
        
        r1 = R[:, 0]
        r3 = R[:, 2]
        H = np.zeros((2, 15), dtype=float)
        H[0, 3:6] = r1
        H[1, 3:6] = r3
        v_skew = np.array([
            [0.0, -ekf.v_enu[2], ekf.v_enu[1]],
            [ekf.v_enu[2], 0.0, -ekf.v_enu[0]],
            [-ekf.v_enu[1], ekf.v_enu[0], 0.0]
        ])
        H[0, 6:9] = -(r1 @ v_skew)
        H[1, 6:9] = -(r3 @ v_skew)
        
        sigma_vert = ekf.compute_nhc_sigma_vert()
        R_meas = np.diag([sigma_lat**2, sigma_vert**2])
        ekf._apply_kalman_update(H, z, R_meas)
        
    lat, lon, _ = ekf.get_position_lat_lon()
    return compute_final_error(lat, lon)

print("Baseline References:")
print(f"  EKF v1 (no curvature):            {run_custom_ekf(0.0):.1f}m (C++ leaderboard: 1279.2m)")
print(f"  EKF v2 (accel klat=5, kyaw=2):     {run_custom_ekf(0.0, klat=5.0, kyaw=2.0):.1f}m (C++ leaderboard: 749.7m)")
print(f"  EKF v3 (kc=2.0, gyro_z):           {run_custom_ekf(2.0, gyro_mode='z'):.1f}m (C++ leaderboard: 1298.7m)")

print("\n--- Test 1: Gyro Axis Mode (with kc=2.0, estimated speed) ---")
for mode in ["z", "y", "norm", "nav_z"]:
    err = run_custom_ekf(2.0, gyro_mode=mode)
    print(f"  gyro_mode={mode:6s}: Error = {err:7.1f}m")

print("\n--- Test 2: Gyro Axis Mode (with kc=2.0, GT speed) ---")
for mode in ["z", "y", "norm", "nav_z"]:
    err = run_custom_ekf(2.0, use_gt_speed=True, gyro_mode=mode)
    print(f"  gyro_mode={mode:6s} + GT speed: Error = {err:7.1f}m")

print("\n--- Test 3: kc Sweep with gyro_mode='y' (which carries the turn) ---")
for kc in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    err = run_custom_ekf(kc, gyro_mode='y')
    print(f"  kc={kc:4.1f} (gyro_y): Error = {err:7.1f}m")

print("\n--- Test 4: kc Sweep with gyro_mode='norm' (axis-independent!) ---")
for kc in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    err = run_custom_ekf(kc, gyro_mode='norm')
    print(f"  kc={kc:4.1f} (gyro_norm): Error = {err:7.1f}m")

