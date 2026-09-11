import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/outages/train/pair_S3c__120s__0.parquet')
mask = (df['timestamp_s'] >= 2943.8 + 50) & (df['timestamp_s'] <= 2943.8 + 75)
sub = df[mask].copy()
sub['t_rel'] = sub['timestamp_s'] - 2943.8
for _, r in sub.iloc[::20].iterrows():
    can_hdg = r["can_gps_heading_deg"]
    can_yaw = r["can_yaw_rate_deg_s"]
    gx = np.degrees(r["phone_gyro_x_rad_s"])
    gy = np.degrees(r["phone_gyro_y_rad_s"])
    gz = np.degrees(r["phone_gyro_z_rad_s"])
    print(f"t={r['t_rel']:5.1f}s | CAN hdg={can_hdg:6.1f}° | CAN yaw={can_yaw:5.2f}°/s | gx={gx:+5.2f}°, gy={gy:+5.2f}°, gz={gz:+5.2f}°")
