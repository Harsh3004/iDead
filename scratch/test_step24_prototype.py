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

class EkfV4Prototype(ErrorStateEkf):
    def __init__(self, config=None, kc=2.0, cutoff_hz=2.0, use_speed_floor=True, use_nav_yaw=True):
        super().__init__(config)
        self.kc = kc
        self.cutoff_hz = cutoff_hz
        self.use_speed_floor = use_speed_floor
        self.use_nav_yaw = use_nav_yaw
        self.v0 = 0.0
        self.omega_yaw_filt = 0.0
        self.last_t_lpf = None

    def predict(self, t, f_body, omega_body):
        super().predict(t, f_body, omega_body)
        
        # Compute horizontal yaw rate
        w_unbiased = self.last_omega_body - self.b_gyro
        if self.use_nav_yaw:
            R = quat_to_rot_matrix(self.q)
            w_nav = R @ w_unbiased
            w_yaw_raw = abs(w_nav[2])
        else:
            w_yaw_raw = abs(w_unbiased[2])
            
        # Update first-order low-pass filter
        if self.last_t_lpf is None:
            self.omega_yaw_filt = w_yaw_raw
            self.last_t_lpf = t
        else:
            dt = t - self.last_t_lpf
            if dt > 0.0 and dt <= 1.0:
                tau = 1.0 / (2.0 * np.pi * self.cutoff_hz) if self.cutoff_hz > 1e-4 else 0.0
                alpha = dt / (dt + tau) if tau > 1e-6 else 1.0
                self.omega_yaw_filt += alpha * (w_yaw_raw - self.omega_yaw_filt)
            self.last_t_lpf = t

    def compute_nhc_sigma_lat(self) -> float:
        dt_init = max(0.0, self.t - self.t0)
        settle_extra = 0.0
        if self.cfg.nhc_settle_duration_s > 1e-6:
            settle_extra = self.cfg.nhc_settle_sigma_extra * np.exp(-dt_init / self.cfg.nhc_settle_duration_s)

        speed = self.get_speed_ms()
        v_curv = max(speed, self.v0) if self.use_speed_floor else speed

        a_c = v_curv * self.omega_yaw_filt
        c_term = self.kc * a_c
        curv_scale = np.sqrt(1.0 + c_term * c_term)
        return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

def simulate_v4(outage_id, kc=2.0, cutoff_hz=2.0, use_speed_floor=True, use_nav_yaw=True):
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{outage_id}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)

    run_id = outage_id.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None

    cfg = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0)
    ekf = EkfV4Prototype(cfg, kc=kc, cutoff_hz=cutoff_hz, use_speed_floor=use_speed_floor, use_nav_yaw=use_nav_yaw)
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

print("=== STEP 24 PROTOTYPE VALIDATION ===")
# 1. Anchor case pair_S3c__120s__0
print("\n--- Anchor Case: pair_S3c__120s__0 ---")
print("  v2 Error: 749.7m")
print("  v3 Error: 1298.7m")
for kc in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
    err = simulate_v4("pair_S3c__120s__0", kc=kc, cutoff_hz=2.0)
    print(f"  v4 (kc={kc:.1f}, cutoff=2.0Hz): {err:.1f}m")

# 2. Straight road vibration immunity
print("\n--- Straight Road Vibration Immunity Check ---")
for oid in ["pair_S1__180s__0", "pair_S3b__180s__0", "pair_Vta1a__60s__0", "pair_Vta17__180s__0"]:
    err_v4 = simulate_v4(oid, kc=3.0, cutoff_hz=2.0)
    print(f"  {oid:20s}: v4 error = {err_v4:.1f}m")
