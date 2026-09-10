"""Module B v1: Dynamic-Phase Yaw Estimation and Attitude Quaternion Synthesis.

Estimates initial heading (yaw) via Principal Component Analysis (PCA) on early
trajectory motion, performs a gyro-Z angular rate integration cross-check,
and combines dynamic yaw with Step 14's static leveling and gyro bias into
a unified initial attitude quaternion.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


# Default thresholds for dynamic motion detection
DEFAULT_MOTION_SPEED_THRESH_KMH: float = 5.0
DEFAULT_MIN_MOTION_DURATION_S: float = 3.0
DEFAULT_WINDOW_LEN_S: float = 5.0
MEAN_EARTH_RADIUS_M: float = 6371000.0


@dataclass(frozen=True)
class MotionWindow:
    """Represents a detected contiguous motion time window."""
    start_s: float
    end_s: float
    duration_s: float
    sample_count: int
    mean_speed_kmh: float


@dataclass(frozen=True)
class DynamicYawResult:
    """Result of PCA heading estimation and gyro cross-check."""
    run_id: str
    has_motion: bool
    start_s: float
    end_s: float
    duration_s: float
    yaw_rad: float
    yaw_deg: float
    yaw_source: str  # 'pca_early_motion' or 'none_parked'
    pca_heading_change_deg: float
    gyro_heading_change_deg: float
    discrepancy_deg: float  # pca_heading_change - gyro_heading_change
    displacement_m: float
    flag: str


@dataclass(frozen=True)
class InitialAttitudeResult:
    """Full synthesized initial attitude and calibration for a run."""
    run_id: str
    has_motion: bool
    yaw_rad: float
    yaw_deg: float
    yaw_source: str
    pitch_rad: float
    roll_rad: float
    pitch_deg: float
    roll_deg: float
    leveling_source: str  # 'module_a_measured' or 'fallback_zero'
    gyro_bias_x_rad_s: float
    gyro_bias_y_rad_s: float
    gyro_bias_z_rad_s: float
    gyro_bias_source: str  # 'module_a_measured' or 'fallback_zero'
    q_w: float
    q_x: float
    q_y: float
    q_z: float
    pca_heading_change_deg: float
    gyro_heading_change_deg: float
    discrepancy_deg: float
    flag: str


def quat_multiply(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Hamilton quaternion product q1 * q2, where q = (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return w, x, y, z


def quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Normalize quaternion to unit length."""
    w, x, y, z = q
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    return w / norm, x / norm, y / norm, z / norm


def construct_attitude_quaternion(
    yaw_rad: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0
) -> Tuple[float, float, float, float]:
    """Construct attitude quaternion from yaw, pitch, roll.

    Conventions matching core/include/idr/quaternion.hpp:
      - Navigation frame: Local Tangent Plane East-North-Up (ENU)
      - Body frame: X = Right, Y = Forward, Z = Up
      - Yaw (psi): Azimuth clockwise from North.
          q_yaw = [cos(psi/2), 0, 0, -sin(psi/2)]  (rotation about -Z_nav)
      - Pitch (theta): Nose-up rotation about body X.
          q_pitch = [cos(theta/2), sin(theta/2), 0, 0]
      - Roll (phi): Right-side-down rotation about body Y.
          q_roll = [cos(phi/2), 0, sin(phi/2), 0]
      - Combined: q = q_yaw * q_pitch * q_roll
    """
    if np.isnan(yaw_rad):
        return np.nan, np.nan, np.nan, np.nan

    p = 0.0 if np.isnan(pitch_rad) else pitch_rad
    r = 0.0 if np.isnan(roll_rad) else roll_rad

    half_psi = 0.5 * yaw_rad
    q_yaw = (float(np.cos(half_psi)), 0.0, 0.0, float(-np.sin(half_psi)))

    half_pitch = 0.5 * p
    q_pitch = (float(np.cos(half_pitch)), float(np.sin(half_pitch)), 0.0, 0.0)

    half_roll = 0.5 * r
    q_roll = (float(np.cos(half_roll)), 0.0, float(np.sin(half_roll)), 0.0)

    q_yp = quat_multiply(q_yaw, q_pitch)
    q_full = quat_multiply(q_yp, q_roll)
    return quat_normalize(q_full)


def detect_early_motion_window(
    df: pd.DataFrame,
    min_speed_kmh: float = DEFAULT_MOTION_SPEED_THRESH_KMH,
    min_duration_s: float = DEFAULT_MIN_MOTION_DURATION_S,
    window_len_s: float = DEFAULT_WINDOW_LEN_S,
) -> Optional[MotionWindow]:
    """Identify the first sustained motion window after recording starts.

    Criteria:
      1. Speed (CAN indicated speed if present, else phone GPS speed) >= min_speed_kmh
      2. Sustained contiguous duration >= min_duration_s
      3. Window length capped at window_len_s (e.g. 5.0 seconds = 50 samples)
    """
    if len(df) < 5:
        return None

    spd_col = "can_indicated_vehicle_speed_kmh" if "can_indicated_vehicle_speed_kmh" in df.columns else "phone_gps_speed_kmh"
    if spd_col not in df.columns:
        return None

    is_moving = df[spd_col] >= min_speed_kmh
    if not is_moving.any():
        return None

    blocks = (is_moving != is_moving.shift()).cumsum()

    for _, group in df[is_moving].groupby(blocks):
        t0 = float(group["timestamp_s"].iloc[0])
        t1 = float(group["timestamp_s"].iloc[-1])
        dur = t1 - t0
        if dur >= min_duration_s:
            # Take first window_len_s
            eff = group[group["timestamp_s"] <= (t0 + window_len_s)]
            eff_start = float(eff["timestamp_s"].iloc[0])
            eff_end = float(eff["timestamp_s"].iloc[-1])
            return MotionWindow(
                start_s=eff_start,
                end_s=eff_end,
                duration_s=eff_end - eff_start,
                sample_count=len(eff),
                mean_speed_kmh=float(eff[spd_col].mean()),
            )

    return None


def compute_pca_heading(
    lat_series: pd.Series,
    lon_series: pd.Series
) -> Tuple[float, float, float]:
    """Estimate 2D trajectory heading using Principal Component Analysis (PCA).

    Projects coordinates to local tangent plane East-North (meters), computes
    the 2x2 covariance matrix, and extracts the primary eigenvector. The sign
    is resolved using the forward displacement vector.

    Returns:
        (heading_rad, heading_deg, total_displacement_m)
        heading_rad: in [0, 2*pi) radians clockwise from North
        heading_deg: in [0, 360.0) degrees clockwise from North
        total_displacement_m: straight-line distance from start to end in meters.
    """
    if len(lat_series) < 2:
        return np.nan, np.nan, 0.0

    lat0 = float(lat_series.iloc[0])
    lon0 = float(lon_series.iloc[0])

    lat_rad = np.deg2rad(lat_series.values)
    lon_rad = np.deg2rad(lon_series.values)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    e = MEAN_EARTH_RADIUS_M * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    n = MEAN_EARTH_RADIUS_M * (lat_rad - lat0_rad)

    pts = np.column_stack([e, n])
    disp = pts[-1] - pts[0]
    dist = float(np.linalg.norm(disp))

    if dist < 0.5:
        # Stationary or insufficient displacement
        return np.nan, np.nan, dist

    pts_centered = pts - pts.mean(axis=0)
    cov = np.cov(pts_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]

    # Resolve sign ambiguity: forward progression
    if float(np.dot(disp, v)) < 0.0:
        v = -v

    heading_rad = float(np.arctan2(v[0], v[1])) % (2.0 * np.pi)
    heading_deg = float(np.rad2deg(heading_rad)) % 360.0
    return heading_rad, heading_deg, dist


def integrate_gyro_heading_change(
    time_series: pd.Series,
    gyro_z_series: pd.Series
) -> float:
    """Integrate body gyro-Z to determine heading change over interval.

    In Body (X=Right, Y=Forward, Z=Up) and ENU (heading clockwise from North):
    Turning left is positive gyro-Z (counter-clockwise about Up), which
    decreases heading: d_psi/dt = -omega_z.
    Turning right is negative gyro-Z, which increases heading.
    Therefore:
      Delta_psi = - integral(omega_z * dt)

    Returns:
        delta_heading_deg: heading change in degrees.
    """
    if len(time_series) < 2:
        return 0.0

    dt = np.diff(time_series.values)
    gz = gyro_z_series.values[:-1]  # or trapezoidal
    delta_rad = -float(np.sum(gz * dt))
    return float(np.rad2deg(delta_rad))


def compute_heading_discrepancy(
    lat_series: pd.Series,
    lon_series: pd.Series,
    time_series: pd.Series,
    gyro_z_series: pd.Series,
) -> Tuple[float, float, float]:
    """Compute trajectory heading change vs integrated gyro heading change.

    Divides window into two halves (first half and second half), computes
    PCA heading on each half, and compares delta_psi_pca with delta_psi_gyro.

    Returns:
        (pca_delta_deg, gyro_delta_deg, discrepancy_deg)
        where discrepancy_deg = pca_delta_deg - gyro_delta_deg
    """
    n = len(lat_series)
    if n < 6:
        return np.nan, np.nan, np.nan

    half = n // 2
    h_start_rad, h_start_deg, d1 = compute_pca_heading(lat_series.iloc[:half], lon_series.iloc[:half])
    h_end_rad, h_end_deg, d2 = compute_pca_heading(lat_series.iloc[half:], lon_series.iloc[half:])

    if np.isnan(h_start_deg) or np.isnan(h_end_deg):
        # Fall back to overall displacement bearing difference if half-slices too short
        disp_start = np.array([
            MEAN_EARTH_RADIUS_M * np.cos(np.deg2rad(lat_series.iloc[0])) * np.deg2rad(lon_series.iloc[half] - lon_series.iloc[0]),
            MEAN_EARTH_RADIUS_M * np.deg2rad(lat_series.iloc[half] - lat_series.iloc[0]),
        ])
        disp_end = np.array([
            MEAN_EARTH_RADIUS_M * np.cos(np.deg2rad(lat_series.iloc[half])) * np.deg2rad(lon_series.iloc[-1] - lon_series.iloc[half]),
            MEAN_EARTH_RADIUS_M * np.deg2rad(lat_series.iloc[-1] - lat_series.iloc[half]),
        ])
        if np.linalg.norm(disp_start) > 0.5 and np.linalg.norm(disp_end) > 0.5:
            h_start_deg = float(np.rad2deg(np.arctan2(disp_start[0], disp_start[1]))) % 360.0
            h_end_deg = float(np.rad2deg(np.arctan2(disp_end[0], disp_end[1]))) % 360.0
        else:
            return np.nan, np.nan, np.nan

    # Circular difference in [-180, 180]
    pca_delta_deg = (h_end_deg - h_start_deg + 180.0) % 360.0 - 180.0

    # Effective time of the first half heading is the centroid of the first half timestamps
    # Effective time of the second half heading is the centroid of the second half timestamps
    t_c1 = float(time_series.iloc[:half].mean())
    t_c2 = float(time_series.iloc[half:].mean())
    mask_gyro = (time_series >= t_c1) & (time_series <= t_c2)
    if mask_gyro.sum() >= 2:
        gyro_delta_deg = integrate_gyro_heading_change(time_series[mask_gyro], gyro_z_series[mask_gyro])
    else:
        gyro_delta_deg = integrate_gyro_heading_change(time_series, gyro_z_series)

    discrepancy_deg = pca_delta_deg - gyro_delta_deg
    return pca_delta_deg, gyro_delta_deg, discrepancy_deg


def estimate_dynamic_phase(
    df: pd.DataFrame,
    run_id: str,
    static_record: Optional[Dict[str, float]] = None,
    min_speed_kmh: float = DEFAULT_MOTION_SPEED_THRESH_KMH,
    min_duration_s: float = DEFAULT_MIN_MOTION_DURATION_S,
    window_len_s: float = DEFAULT_WINDOW_LEN_S,
) -> InitialAttitudeResult:
    """Run end-to-end dynamic yaw estimation and synthesize attitude quaternion.

    Args:
        df: Paired DataFrame containing position, speed, and gyro channels.
        run_id: Identifier string for the run.
        static_record: Optional dictionary containing Step 14 results (pitch_rad, roll_rad, gyro_biases).
        min_speed_kmh: Speed threshold for motion detection.
        min_duration_s: Minimum duration for motion window.
        window_len_s: Length of early motion window to analyze.

    Returns:
        InitialAttitudeResult with synthesized quaternion and calibration sources.
    """
    motion = detect_early_motion_window(
        df,
        min_speed_kmh=min_speed_kmh,
        min_duration_s=min_duration_s,
        window_len_s=window_len_s,
    )

    # Position column selection: CAN GNSS is primary reference; fallback to phone GPS
    lat_col = "can_latitude_deg" if "can_latitude_deg" in df.columns else "phone_latitude_deg"
    lon_col = "can_longitude_deg" if "can_longitude_deg" in df.columns else "phone_longitude_deg"

    # Static inputs from Step 14 if available
    pitch_rad = np.nan
    roll_rad = np.nan
    pitch_deg = np.nan
    roll_deg = np.nan
    leveling_source = "fallback_zero"
    gb_x = 0.0
    gb_y = 0.0
    gb_z = 0.0
    gyro_bias_source = "fallback_zero"

    if static_record and static_record.get("window_found", False):
        p_val = static_record.get("pitch_rad")
        r_val = static_record.get("roll_rad")
        if not np.isnan(p_val) and not np.isnan(r_val):
            pitch_rad = float(p_val)
            roll_rad = float(r_val)
            pitch_deg = float(np.rad2deg(pitch_rad))
            roll_deg = float(np.rad2deg(roll_rad))
            leveling_source = "module_a_measured"

        bx = static_record.get("gyro_bias_x_rad_s")
        by = static_record.get("gyro_bias_y_rad_s")
        bz = static_record.get("gyro_bias_z_rad_s")
        if not np.isnan(bx) and not np.isnan(by) and not np.isnan(bz):
            gb_x = float(bx)
            gb_y = float(by)
            gb_z = float(bz)
            gyro_bias_source = "module_a_measured"

    if motion is None:
        # Fully stationary run (e.g. pair_Vw1, pair_Vw15)
        return InitialAttitudeResult(
            run_id=run_id,
            has_motion=False,
            yaw_rad=np.nan,
            yaw_deg=np.nan,
            yaw_source="none_parked",
            pitch_rad=pitch_rad if leveling_source == "module_a_measured" else 0.0,
            roll_rad=roll_rad if leveling_source == "module_a_measured" else 0.0,
            pitch_deg=pitch_deg if leveling_source == "module_a_measured" else 0.0,
            roll_deg=roll_deg if leveling_source == "module_a_measured" else 0.0,
            leveling_source=leveling_source,
            gyro_bias_x_rad_s=gb_x,
            gyro_bias_y_rad_s=gb_y,
            gyro_bias_z_rad_s=gb_z,
            gyro_bias_source=gyro_bias_source,
            q_w=np.nan,
            q_x=np.nan,
            q_y=np.nan,
            q_z=np.nan,
            pca_heading_change_deg=0.0,
            gyro_heading_change_deg=0.0,
            discrepancy_deg=0.0,
            flag="no_motion_found",
        )

    # Slice early motion window
    w_df = df[(df["timestamp_s"] >= motion.start_s) & (df["timestamp_s"] <= motion.end_s)]

    yaw_rad, yaw_deg, dist = compute_pca_heading(w_df[lat_col], w_df[lon_col])

    # Gyro-Z cross check
    pca_delta_deg, gyro_delta_deg, discrepancy_deg = compute_heading_discrepancy(
        w_df[lat_col],
        w_df[lon_col],
        w_df["timestamp_s"],
        w_df["phone_gyro_z_rad_s"],
    )

    # Attitude Quaternion construction (using 0.0 fallback for pitch/roll if not measured)
    effective_pitch = pitch_rad if leveling_source == "module_a_measured" else 0.0
    effective_roll = roll_rad if leveling_source == "module_a_measured" else 0.0
    qw, qx, qy, qz = construct_attitude_quaternion(yaw_rad, effective_pitch, effective_roll)

    flag = "clean_motion"
    if abs(discrepancy_deg) > 20.0:
        flag = "high_gyro_discrepancy"
    elif leveling_source == "fallback_zero":
        flag = "uncalibrated_leveling"

    return InitialAttitudeResult(
        run_id=run_id,
        has_motion=True,
        yaw_rad=yaw_rad,
        yaw_deg=yaw_deg,
        yaw_source="pca_early_motion",
        pitch_rad=effective_pitch,
        roll_rad=effective_roll,
        pitch_deg=float(np.rad2deg(effective_pitch)),
        roll_deg=float(np.rad2deg(effective_roll)),
        leveling_source=leveling_source,
        gyro_bias_x_rad_s=gb_x,
        gyro_bias_y_rad_s=gb_y,
        gyro_bias_z_rad_s=gb_z,
        gyro_bias_source=gyro_bias_source,
        q_w=qw,
        q_x=qx,
        q_y=qy,
        q_z=qz,
        pca_heading_change_deg=round(pca_delta_deg, 3) if not np.isnan(pca_delta_deg) else 0.0,
        gyro_heading_change_deg=round(gyro_delta_deg, 3) if not np.isnan(gyro_delta_deg) else 0.0,
        discrepancy_deg=round(discrepancy_deg, 3) if not np.isnan(discrepancy_deg) else 0.0,
        flag=flag,
    )
