"""Test diagnostic hypotheses for curvature adaptive NHC to understand the failure modes."""

import numpy as np
import pandas as pd
from pathlib import Path
from dataeval.estimation.ekf import ErrorStateEkf, EkfConfig, quat_normalize, quat_multiply
from dataeval.harness.metrics import compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

def parse_header(line):
    import re
    m = re.findall(r'(\w+)=([-+]?[0-9]*\.?[0-9]+)', line)
    return {k: float(v) for k, v in m}

att_df = pd.read_csv("results/module_b_initial_attitude.csv", comment="#").set_index("run_id")
manifest = pd.read_csv("data/processed/outages/manifest.csv").set_index("outage_id")

def from_euler_enu(yaw_deg, pitch_rad, roll_rad):
    half_yaw = 0.5 * np.radians(yaw_deg)
    q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, -np.sin(half_yaw)])
    half_pitch = 0.5 * pitch_rad
    q_pitch = np.array([np.cos(half_pitch), np.sin(half_pitch), 0.0, 0.0])
    half_roll = 0.5 * roll_rad
    q_roll = np.array([np.cos(half_roll), 0.0, np.sin(half_roll), 0.0])
    return quat_normalize(quat_multiply(quat_multiply(q_yaw, q_pitch), q_roll))

class HypoEkf(ErrorStateEkf):
    """Custom subclass to test candidate curvature calculation hypotheses."""
    def __init__(self, config=None, hypo_mode="v2", deadband=0.8, ema_alpha=0.05):
        super().__init__(config)
        self.hypo_mode = hypo_mode
        self.deadband = deadband
        self.ema_alpha = ema_alpha
        self.filtered_a_lat = 0.0

    def compute_nhc_sigma_lat(self) -> float:
        dt_init = max(0.0, self.t - self.t0)
        settle_extra = 0.0
        if self.cfg.nhc_settle_duration_s > 1e-6:
            settle_extra = self.cfg.nhc_settle_sigma_extra * np.exp(-dt_init / self.cfg.nhc_settle_duration_s)

        raw_a_lat = abs(self.last_f_body[0] - self.b_accel[0])
        w_yaw = abs(self.last_omega_body[2] - self.b_gyro[2])
        speed = self.get_speed_ms()

        if self.hypo_mode == "v1":
            return float(self.cfg.sigma_nhc_lat)

        elif self.hypo_mode == "settle_only":
            return float(self.cfg.sigma_nhc_lat + settle_extra)

        elif self.hypo_mode == "v2_raw":
            lat_term = self.cfg.nhc_curv_lat_coeff * raw_a_lat
            yaw_term = self.cfg.nhc_curv_yaw_coeff * w_yaw
            curv_scale = np.sqrt(1.0 + lat_term * lat_term + yaw_term * yaw_term)
            return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

        elif self.hypo_mode == "deadband":
            # Only trigger if raw_a_lat exceeds deadband
            eff_a_lat = max(0.0, raw_a_lat - self.deadband)
            lat_term = self.cfg.nhc_curv_lat_coeff * eff_a_lat
            yaw_term = self.cfg.nhc_curv_yaw_coeff * w_yaw
            curv_scale = np.sqrt(1.0 + lat_term * lat_term + yaw_term * yaw_term)
            return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

        elif self.hypo_mode == "ema_filtered":
            # Low-pass filter a_lat
            self.filtered_a_lat = (1.0 - self.ema_alpha) * self.filtered_a_lat + self.ema_alpha * raw_a_lat
            lat_term = self.cfg.nhc_curv_lat_coeff * self.filtered_a_lat
            yaw_term = self.cfg.nhc_curv_yaw_coeff * w_yaw
            curv_scale = np.sqrt(1.0 + lat_term * lat_term + yaw_term * yaw_term)
            return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

        elif self.hypo_mode == "kinematic_centripetal":
            # a_c = speed * w_yaw
            a_c = speed * w_yaw
            # Pure kinematic centripetal acceleration (gyro-based, immune to vibration)
            curv_scale = np.sqrt(1.0 + (self.cfg.nhc_curv_lat_coeff * a_c) ** 2)
            return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

        elif self.hypo_mode == "capped_scale":
            lat_term = self.cfg.nhc_curv_lat_coeff * raw_a_lat
            yaw_term = self.cfg.nhc_curv_yaw_coeff * w_yaw
            curv_scale = min(3.0, np.sqrt(1.0 + lat_term * lat_term + yaw_term * yaw_term))
            return float((self.cfg.sigma_nhc_lat + settle_extra) * curv_scale)

        return float(self.cfg.sigma_nhc_lat)

def run_hypo_sim(outage_id, hypo_mode, **kwargs):
    cache_csv = Path(f"data/processed/_cpp_replay_cache/{outage_id}.csv")
    with open(cache_csv, "r") as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    
    run_id = outage_id.split("__")[0]
    att = att_df.loc[run_id] if run_id in att_df.index else None
    
    cfg = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0, nhc_curv_lat_coeff=5.0, nhc_curv_yaw_coeff=2.0)
    ekf = HypoEkf(config=cfg, hypo_mode=hypo_mode, **kwargs)
    
    if att is not None and pd.notna(att.get("q_w")):
        q0 = np.array([att["q_w"], att["q_x"], att["q_y"], att["q_z"]])
    elif att is not None and att.get("leveling_source") == "module_a_measured":
        q0 = from_euler_enu(init["heading_deg"], att["pitch_rad"], att["roll_rad"])
    else:
        half_psi = 0.5 * np.radians(init["heading_deg"])
        q0 = np.array([np.cos(half_psi), 0.0, 0.0, -np.sin(half_psi)])
        
    gyro_bias = np.zeros(3)
    if att is not None and att.get("gyro_bias_source") == "module_a_measured":
        gyro_bias = np.array([att["gyro_bias_x_rad_s"], att["gyro_bias_y_rad_s"], att["gyro_bias_z_rad_s"]])
        
    ekf.initialize(
        t0=init["t0"],
        lat0=init["lat0"],
        lon0=init["lon0"],
        alt0=init["alt0"],
        speed_ms=init["speed_ms"],
        heading_deg=init["heading_deg"],
        q0=q0,
        b_gyro0=gyro_bias,
    )
    
    preds = []
    for _, row in df_imu.iterrows():
        t = row["timestamp_s"]
        f_b = np.array([row["ax"], row["ay"], row["az"]])
        omega_b = np.array([row["gx"], row["gy"], row["gz"]])
        
        ekf.predict(t, f_b, omega_b)
        if ekf.is_stationary():
            ekf.update_zupt()
        ekf.update_nhc()
        
        lat, lon, alt = ekf.get_position_lat_lon()
        preds.append({
            "timestamp_s": t,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "speed_ms": ekf.get_speed_ms(),
            "heading_deg": ekf.get_heading_deg(),
        })
        
    pred_df = pd.DataFrame(preds)
    man = manifest.loc[outage_id]
    outage_parquet = Path(f"data/processed/outages/{man['split']}/{outage_id}.parquet")
    df_parquet = pd.read_parquet(outage_parquet)
    window = OutageWindow(
        run_id=man["run_or_pair_id"],
        source_side=man["source_side"],
        start_s=man["start_s"],
        duration_s=float(man["outage_s"]),
        end_s=man["end_s"],
    )
    gt_df = ground_truth_trajectory(df_parquet, window)
    metrics = compute_outage_metrics(pred_df, gt_df)
    return metrics.final_pos_error_m

test_instances = [
    # Top regressed instances
    "pair_S1__180s__0",
    "pair_Vtb1__180s__0",
    "pair_S3b__180s__0",
    "pair_Vta17__180s__0",
    "pair_Vta1a__60s__0",
    "pair_Vw4__30s__0",
    # Target case
    "pair_S3c__180s__0",
    "pair_S3c__120s__0",
    "pair_Vw16b__10s__0",
]

modes = ["v1", "settle_only", "v2_raw", "deadband", "ema_filtered", "kinematic_centripetal"]

print(f"{'Outage ID':20s} | {'v1 (Base)':10s} | {'SettleOnly':10s} | {'v2 (RawCurv)':12s} | {'Deadband(1m)':12s} | {'EMA-Filtered':12s} | {'Kinematic(v*w)':14s}")
print("-" * 105)

for oid in test_instances:
    e_v1 = run_hypo_sim(oid, "v1")
    e_set = run_hypo_sim(oid, "settle_only")
    e_v2 = run_hypo_sim(oid, "v2_raw")
    e_db = run_hypo_sim(oid, "deadband", deadband=1.0)
    e_ema = run_hypo_sim(oid, "ema_filtered", ema_alpha=0.02)
    e_kin = run_hypo_sim(oid, "kinematic_centripetal")
    
    print(f"{oid:20s} | {e_v1:8.2f}m  | {e_set:8.2f}m  | {e_v2:8.2f}m     | {e_db:8.2f}m     | {e_ema:8.2f}m     | {e_kin:8.2f}m")
