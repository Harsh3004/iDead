import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_to_rot_matrix
from dataeval.harness.metrics import compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
import re

def parse_header(line):
    m = re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line)
    return {k: float(v) for k, v in m}

att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#").set_index("run_id")
manifest = pd.read_csv("data/processed/outages/manifest.csv").set_index("outage_id")

class CustomEkf(ErrorStateEkf):
    def __init__(self, config=None, kc=2.0, gyro_mode="z", speed_mode="est", klat=0.0, kyaw=0.0):
        super().__init__(config)
        self.kc = kc
        self.gyro_mode = gyro_mode
        self.speed_mode = speed_mode
        self.klat = klat
        self.kyaw = kyaw
        self.v0 = 0.0

    def compute_nhc_sigma_lat(self) -> float:
        dt_init = max(0.0, self.t - self.t0)
        settle_extra = 0.0
        if self.cfg.nhc_settle_duration_s > 1e-6:
            settle_extra = self.cfg.nhc_settle_sigma_extra * np.exp(-dt_init / self.cfg.nhc_settle_duration_s)

        # Speed
        if self.speed_mode == "est":
            speed = self.get_speed_ms()
        elif self.speed_mode == "v0":
            speed = self.v0
        elif self.speed_mode == "max_v0":
            speed = max(self.get_speed_ms(), self.v0)
        elif self.speed_mode == "floor5":
            speed = max(self.get_speed_ms(), 5.0)
        else:
            speed = self.get_speed_ms()

        # Gyro
        w_unbiased = self.last_omega_body - self.b_gyro
        if self.gyro_mode == "z":
            w_yaw = abs(w_unbiased[2])
        elif self.gyro_mode == "norm":
            w_yaw = float(np.linalg.norm(w_unbiased))
        elif self.gyro_mode == "nav_z":
            R = quat_to_rot_matrix(self.q)
            w_nav = R @ w_unbiased
            w_yaw = abs(w_nav[2])
        elif self.gyro_mode == "y":
            w_yaw = abs(w_unbiased[1])
        else:
            w_yaw = abs(w_unbiased[2])

        a_c = speed * w_yaw
        c_term = self.kc * a_c
        yaw_term = self.kyaw * w_yaw

        lat_term = 0.0
        if self.klat > 1e-6:
            a_lat = abs(self.last_f_body[0] - self.b_accel[0])
            lat_term = self.klat * a_lat

        curv_scale = np.sqrt(1.0 + c_term * c_term + lat_term * lat_term + yaw_term * yaw_term)
        return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

def simulate(outage_id, kc=2.0, gyro_mode="z", speed_mode="est", klat=0.0, kyaw=0.0):
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{outage_id}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)

    run_id = outage_id.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None

    cfg = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0)
    ekf = CustomEkf(cfg, kc=kc, gyro_mode=gyro_mode, speed_mode=speed_mode, klat=klat, kyaw=kyaw)
    ekf.v0 = init["speed_ms"]

    q0 = np.array([att["q_w"], att["q_x"], att["q_y"], att["q_z"]])
    gb0 = np.array([att["gyro_bias_x_rad_s"], att["gyro_bias_y_rad_s"], att["gyro_bias_z_rad_s"]])
    init_accel = np.array([init["ax0"], init["ay0"], init["az0"]])

    ekf.initialize(
        t0=init["t0"], lat0=init["lat0"], lon0=init["lon0"], alt0=init["alt0"],
        speed_ms=init["speed_ms"], heading_deg=init["heading_deg"],
        q0=q0, b_gyro0=gb0, b_accel0=np.zeros(3), initial_accel=init_accel
    )

    preds = []
    for _, row in df_imu.iterrows():
        t = row["timestamp_s"]
        f_b = np.array([row["ax"], row["ay"], row["az"]])
        w_b = np.array([row["gx"], row["gy"], row["gz"]])

        ekf.predict(t, f_b, w_b)
        if ekf.is_stationary():
            ekf.update_zupt()
        ekf.update_nhc()

        lat, lon, _ = ekf.get_position_lat_lon()
        preds.append({
            "timestamp_s": t,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "speed_ms": ekf.get_speed_ms(),
            "heading_deg": ekf.get_heading_deg(),
        })

    pred_df = pd.DataFrame(preds)
    man = manifest.loc[outage_id]
    df_parquet = pd.read_parquet(f"data/processed/outages/{man['split']}/{outage_id}.parquet")
    window = OutageWindow(
        run_id=man["run_or_pair_id"], source_side=man["source_side"],
        start_s=man["start_s"], duration_s=float(man["outage_s"]), end_s=man["end_s"]
    )
    gt_df = ground_truth_trajectory(df_parquet, window)
    metrics = compute_outage_metrics(pred_df, gt_df)
    return metrics.final_pos_error_m

print("Running Systematic Hypothesis Testing...")
print(f"Test 1: As-is v3 (kc=2.0, gyro=z, speed=est):        {simulate('pair_S3c__120s__0', kc=2.0, gyro_mode='z', speed_mode='est'):.1f}m")
print(f"Test 2: kc Sweep with as-is formula:")
for kc in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    print(f"  kc={kc:4.1f}: {simulate('pair_S3c__120s__0', kc=kc, gyro_mode='z', speed_mode='est'):.1f}m")

print(f"\nTest 3: Gyro axis fix alone (estimated speed):")
for mode in ['z', 'y', 'norm', 'nav_z']:
    print(f"  gyro_mode={mode:6s}: {simulate('pair_S3c__120s__0', kc=2.0, gyro_mode=mode, speed_mode='est'):.1f}m")

print(f"\nTest 4: Speed fix alone (gyro_mode=z):")
for spd in ['est', 'floor5', 'v0', 'max_v0']:
    print(f"  speed_mode={spd:7s}: {simulate('pair_S3c__120s__0', kc=2.0, gyro_mode='z', speed_mode=spd):.1f}m")

print(f"\nTest 5: Both Gyro axis fix AND Speed fix:")
for mode in ['norm', 'nav_z']:
    for spd in ['floor5', 'v0', 'max_v0']:
        print(f"  gyro={mode:5s} + speed={spd:7s} (kc=2.0): {simulate('pair_S3c__120s__0', kc=2.0, gyro_mode=mode, speed_mode=spd):.1f}m")

print(f"\nTest 6: Vibration Immunity Check on Straight Road (pair_S1__180s__0):")
print(f"  v1 error (baseline):       {simulate('pair_S1__180s__0', kc=0.0, klat=0.0, kyaw=0.0):.1f}m")
print(f"  v2 error (accel vibration): {simulate('pair_S1__180s__0', kc=0.0, klat=5.0, kyaw=2.0):.1f}m")
print(f"  v3 as-is:                   {simulate('pair_S1__180s__0', kc=2.0, gyro_mode='z', speed_mode='est'):.1f}m")
print(f"  Fixed (norm + max_v0, kc=2): {simulate('pair_S1__180s__0', kc=2.0, gyro_mode='norm', speed_mode='max_v0'):.1f}m")
