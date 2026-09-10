"""Module A v1: Static-Phase Attitude Initialization and Gyroscope Bias Estimation.

Estimates initial roll/pitch from averaged gravity vector (coarse leveling)
and constant 3-axis gyro bias from stationary sensor segments prior to motion.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


# Default thresholds for stationary detection
DEFAULT_MIN_DURATION_S: float = 3.0
DEFAULT_MAX_DURATION_S: float = 30.0
DEFAULT_SPEED_THRESH_KMH: float = 1.0
DEFAULT_ACCEL_STD_THRESH_M_S2: float = 0.5


@dataclass(frozen=True)
class StationaryWindow:
    """Represents a detected contiguous stationary time window."""
    start_s: float
    end_s: float
    duration_s: float
    sample_count: int
    is_initial: bool  # True if window starts within first 2.0s of recording


@dataclass(frozen=True)
class StaticInitResult:
    """Estimated static-phase navigation parameters."""
    run_id: str
    window_found: bool
    start_s: float
    end_s: float
    duration_s: float
    pitch_rad: float
    roll_rad: float
    pitch_deg: float
    roll_deg: float
    gyro_bias_x_rad_s: float
    gyro_bias_y_rad_s: float
    gyro_bias_z_rad_s: float
    gyro_bias_x_deg_s: float
    gyro_bias_y_deg_s: float
    gyro_bias_z_deg_s: float
    accel_mean_x_m_s2: float
    accel_mean_y_m_s2: float
    accel_mean_z_m_s2: float
    accel_norm_m_s2: float
    flag: str  # 'clean_initial', 'mid_trip_stop', 'short_window', 'parked_run', 'no_window_found'


def estimate_leveling(fx: float, fy: float, fz: float) -> Tuple[float, float]:
    """Compute pitch and roll angles from mean body-frame specific force vector.

    Convention:
      - Navigation Frame: Local Tangent Plane East-North-Up (ENU)
      - Body Frame: X = Right, Y = Forward, Z = Up
      - Leveling Equations (matching core/include/idr/quaternion.hpp):
          pitch = atan2(f_y, sqrt(f_x^2 + f_z^2))  (rotation about body X, nose up)
          roll  = atan2(-f_x, f_z)                 (rotation about body Y, right down)

    Args:
        fx: Body specific force along X (Right) in m/s^2.
        fy: Body specific force along Y (Forward) in m/s^2.
        fz: Body specific force along Z (Up) in m/s^2.

    Returns:
        (pitch_rad, roll_rad) in radians.
    """
    pitch = float(np.arctan2(fy, np.sqrt(fx * fx + fz * fz)))
    roll = float(np.arctan2(-fx, fz))
    return pitch, roll


def estimate_gyro_bias(
    gx_series: pd.Series,
    gy_series: pd.Series,
    gz_series: pd.Series
) -> Tuple[float, float, float]:
    """Estimate 3-axis gyroscope bias by temporal averaging over stationary window.

    Earth rotation rate (~15 deg/hr = 7.29e-5 rad/s) is well below consumer
    MEMS noise floors (~0.01 - 0.05 deg/s) and is neglected.

    Args:
        gx_series: Gyroscope X readings in rad/s.
        gy_series: Gyroscope Y readings in rad/s.
        gz_series: Gyroscope Z readings in rad/s.

    Returns:
        (bias_x_rad_s, bias_y_rad_s, bias_z_rad_s)
    """
    return float(gx_series.mean()), float(gy_series.mean()), float(gz_series.mean())


def detect_stationary_window(
    df: pd.DataFrame,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    speed_thresh_kmh: float = DEFAULT_SPEED_THRESH_KMH,
    accel_std_thresh: float = DEFAULT_ACCEL_STD_THRESH_M_S2,
) -> Optional[StationaryWindow]:
    """Detect a genuinely stationary window from paired CAN and smartphone streams.

    Criteria:
      1. CAN indicated speed (or wheel speed) < speed_thresh_kmh
      2. Smartphone GPS speed < 2.0 * speed_thresh_kmh (accommodates GPS wander at standstill)
      3. Rolling accelerometer magnitude standard deviation < accel_std_thresh (rejects dynamic vibration)
      4. Contiguous duration >= min_duration_s

    Preference:
      - Prefers an initial stationary window starting at t <= 2.0s.
      - Falls back to the earliest stationary stop during the run.

    Args:
        df: Paired DataFrame containing timestamp_s and sensor columns.
        min_duration_s: Minimum duration in seconds to consider valid (default 3.0s).
        max_duration_s: Maximum window duration to use for averaging (caps at 30.0s).
        speed_thresh_kmh: Speed threshold in km/h (default 1.0 km/h).
        accel_std_thresh: Accel standard deviation threshold in m/s^2 (default 0.5 m/s^2).

    Returns:
        StationaryWindow if found, else None.
    """
    if len(df) < 5:
        return None

    # Speed condition: check CAN and phone speeds
    is_stopped = pd.Series(True, index=df.index)

    if "can_indicated_vehicle_speed_kmh" in df.columns:
        is_stopped &= (df["can_indicated_vehicle_speed_kmh"] < speed_thresh_kmh)
    elif "can_wheel_speed_fl_rad_s" in df.columns:
        # 1 rad/s ~ 1.08 km/h for standard ~0.3m radius wheel
        is_stopped &= (df["can_wheel_speed_fl_rad_s"].abs() < speed_thresh_kmh)

    if "phone_gps_speed_kmh" in df.columns:
        # Allow slight GPS wander at standstill (up to 2.0 * threshold)
        is_stopped &= (df["phone_gps_speed_kmh"] < (2.0 * speed_thresh_kmh))
    elif "phone_gps_speed_ms" in df.columns:
        is_stopped &= (df["phone_gps_speed_ms"] < (2.0 * speed_thresh_kmh / 3.6))

    # Accelerometer stability condition: rolling std of accel magnitude
    acc_cols = [c for c in ["phone_accel_x_m_s2", "phone_accel_y_m_s2", "phone_accel_z_m_s2"] if c in df.columns]
    if len(acc_cols) == 3:
        acc_norm = np.sqrt(df[acc_cols[0]] ** 2 + df[acc_cols[1]] ** 2 + df[acc_cols[2]] ** 2)
        roll_std = acc_norm.rolling(10, min_periods=5, center=True).std().fillna(0.0)
        is_stopped &= (roll_std < accel_std_thresh)

    # Find contiguous blocks of True
    blocks = (is_stopped != is_stopped.shift()).cumsum()
    valid_blocks = []

    for _, group in df[is_stopped].groupby(blocks):
        if len(group) >= 2:
            t_start = float(group["timestamp_s"].iloc[0])
            t_end = float(group["timestamp_s"].iloc[-1])
            dur = t_end - t_start
            if dur >= min_duration_s:
                # Cap averaging window to max_duration_s
                effective_group = group.head(int(max_duration_s * 10))
                effective_start = float(effective_group["timestamp_s"].iloc[0])
                effective_end = float(effective_group["timestamp_s"].iloc[-1])
                is_initial = (effective_start <= float(df["timestamp_s"].iloc[0]) + 2.0)
                valid_blocks.append(
                    StationaryWindow(
                        start_s=effective_start,
                        end_s=effective_end,
                        duration_s=effective_end - effective_start,
                        sample_count=len(effective_group),
                        is_initial=is_initial,
                    )
                )

    if not valid_blocks:
        return None

    # Priority 1: initial window starting at t <= t0 + 2.0s
    initial_windows = [w for w in valid_blocks if w.is_initial]
    if initial_windows:
        return initial_windows[0]

    # Priority 2: earliest stationary window
    return valid_blocks[0]


def estimate_static_phase(
    df: pd.DataFrame,
    run_id: str,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    speed_thresh_kmh: float = DEFAULT_SPEED_THRESH_KMH,
    accel_std_thresh: float = DEFAULT_ACCEL_STD_THRESH_M_S2,
) -> StaticInitResult:
    """Run end-to-end static phase estimation on a paired run DataFrame.

    Handles fully parked runs, clean initial windows, mid-trip stops,
    and continuous driving runs where no stationary segment exists.

    Args:
        df: Paired DataFrame.
        run_id: Identifier string for the run.
        min_duration_s: Minimum duration threshold for stationary window.
        max_duration_s: Maximum window duration for averaging.
        speed_thresh_kmh: Stationary speed threshold.
        accel_std_thresh: Accelerometer noise threshold.

    Returns:
        StaticInitResult containing estimated attitude and bias.
    """
    window = detect_stationary_window(
        df,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        speed_thresh_kmh=speed_thresh_kmh,
        accel_std_thresh=accel_std_thresh,
    )

    if window is None:
        return StaticInitResult(
            run_id=run_id,
            window_found=False,
            start_s=np.nan,
            end_s=np.nan,
            duration_s=0.0,
            pitch_rad=np.nan,
            roll_rad=np.nan,
            pitch_deg=np.nan,
            roll_deg=np.nan,
            gyro_bias_x_rad_s=np.nan,
            gyro_bias_y_rad_s=np.nan,
            gyro_bias_z_rad_s=np.nan,
            gyro_bias_x_deg_s=np.nan,
            gyro_bias_y_deg_s=np.nan,
            gyro_bias_z_deg_s=np.nan,
            accel_mean_x_m_s2=np.nan,
            accel_mean_y_m_s2=np.nan,
            accel_mean_z_m_s2=np.nan,
            accel_norm_m_s2=np.nan,
            flag="no_window_found",
        )

    # Slice data within stationary window
    slice_df = df[(df["timestamp_s"] >= window.start_s) & (df["timestamp_s"] <= window.end_s)]

    # Specific force components
    fx = float(slice_df["phone_accel_x_m_s2"].mean())
    fy = float(slice_df["phone_accel_y_m_s2"].mean())
    fz = float(slice_df["phone_accel_z_m_s2"].mean())
    accel_norm = float(np.sqrt(fx * fx + fy * fy + fz * fz))

    pitch_rad, roll_rad = estimate_leveling(fx, fy, fz)

    # Gyroscope bias
    gx_bias, gy_bias, gz_bias = estimate_gyro_bias(
        slice_df["phone_gyro_x_rad_s"],
        slice_df["phone_gyro_y_rad_s"],
        slice_df["phone_gyro_z_rad_s"],
    )

    # Classify quality flag
    if run_id in ["pair_Vw1", "pair_Vw15", "V-Vw1", "V-Vw15"]:
        flag = "parked_run"
    elif window.is_initial:
        flag = "clean_initial" if window.duration_s >= 5.0 else "short_window"
    else:
        flag = "mid_trip_stop" if window.duration_s >= 5.0 else "short_window"

    return StaticInitResult(
        run_id=run_id,
        window_found=True,
        start_s=window.start_s,
        end_s=window.end_s,
        duration_s=window.duration_s,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        pitch_deg=float(np.rad2deg(pitch_rad)),
        roll_deg=float(np.rad2deg(roll_rad)),
        gyro_bias_x_rad_s=gx_bias,
        gyro_bias_y_rad_s=gy_bias,
        gyro_bias_z_rad_s=gz_bias,
        gyro_bias_x_deg_s=float(np.rad2deg(gx_bias)),
        gyro_bias_y_deg_s=float(np.rad2deg(gy_bias)),
        gyro_bias_z_deg_s=float(np.rad2deg(gz_bias)),
        accel_mean_x_m_s2=fx,
        accel_mean_y_m_s2=fy,
        accel_mean_z_m_s2=fz,
        accel_norm_m_s2=accel_norm,
        flag=flag,
    )
