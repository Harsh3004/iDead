import sys
sys.path.append('.')
from scratch.test_step24_prototype import EkfV4Prototype, parse_header, manifest, att_df
import numpy as np
import pandas as pd
from pathlib import Path
from dataeval.estimation.ekf import EkfConfig
from dataeval.harness.metrics import compute_outage_metrics
from dataeval.harness.outage import OutageWindow, ground_truth_trajectory

class EkfMaxAxis(EkfV4Prototype):
    def predict(self, t, f_body, omega_body):
        super(EkfV4Prototype, self).predict(t, f_body, omega_body)
        w_unbiased = self.last_omega_body - self.b_gyro
        w_yaw_raw = float(np.max(np.abs(w_unbiased)))
        if self.last_t_lpf is None:
            self.omega_yaw_filt = w_yaw_raw
            self.last_t_lpf = t
        else:
            dt = t - self.last_t_lpf
            if dt > 0.0 and dt <= 1.0:
                tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
                alpha = dt / (dt + tau)
                self.omega_yaw_filt += alpha * (w_yaw_raw - self.omega_yaw_filt)
            self.last_t_lpf = t

def sim_max(oid, kc=2.0):
    cache_csv = Path(f'data/processed/_cpp_replay_cache/{oid}.csv')
    with open(cache_csv, 'r') as f:
        line1 = f.readline()
    init = parse_header(line1)
    df_imu = pd.read_csv(cache_csv, skiprows=1)
    run_id = oid.split('__')[0]
    att = att_df.loc[run_id]
    cfg = EkfConfig(nhc_settle_sigma_extra=4.0, nhc_settle_duration_s=2.0)
    ekf = EkfMaxAxis(cfg, kc=kc, cutoff_hz=2.0)
    ekf.v0 = init['speed_ms']
    q0 = np.array([att['q_w'], att['q_x'], att['q_y'], att['q_z']])
    gb0 = np.array([att['gyro_bias_x_rad_s'], att['gyro_bias_y_rad_s'], att['gyro_bias_z_rad_s']])
    init_accel = np.array([init['ax0'], init['ay0'], init['az0']])
    ekf.initialize(init['t0'], init['lat0'], init['lon0'], init['alt0'], init['speed_ms'], init['heading_deg'], q0=q0, b_gyro0=gb0, b_accel0=np.zeros(3), initial_accel=init_accel)
    preds = []
    for _, row in df_imu.iterrows():
        t = row['timestamp_s']
        f_b = np.array([row['ax'], row['ay'], row['az']])
        w_b = np.array([row['gx'], row['gy'], row['gz']])
        ekf.predict(t, f_b, w_b)
        if ekf.is_stationary(): ekf.update_zupt()
        ekf.update_nhc()
        lat, lon, _ = ekf.get_position_lat_lon()
        preds.append({'timestamp_s': t, 'latitude_deg': lat, 'longitude_deg': lon, 'speed_ms': ekf.get_speed_ms(), 'heading_deg': ekf.get_heading_deg()})
    pred_df = pd.DataFrame(preds)
    man = manifest.loc[oid]
    split = man['split']
    df_parquet = pd.read_parquet(f'data/processed/outages/{split}/{oid}.parquet')
    window = OutageWindow(run_id=man['run_or_pair_id'], source_side=man['source_side'], start_s=man['start_s'], duration_s=float(man['outage_s']), end_s=man['end_s'])
    gt_df = ground_truth_trajectory(df_parquet, window)
    metrics = compute_outage_metrics(pred_df, gt_df)
    return metrics.final_pos_error_m

print("Max-axis on anchor pair_S3c__120s__0:")
for kc in [1.0, 2.0, 3.0, 4.0, 5.0]:
    print(f"  kc={kc:.1f}: {sim_max('pair_S3c__120s__0', kc=kc):.1f}m")

print("\nMax-axis on straight runs:")
for oid in ['pair_S1__180s__0', 'pair_S3b__180s__0', 'pair_Vta1a__60s__0', 'pair_Vta17__180s__0']:
    print(f"  {oid:20s}: {sim_max(oid, kc=2.0):.1f}m")
