"""Time-synchronization and grid-alignment harness for CAN and Smartphone paired runs.

This module aligns heterogeneous multi-sensor streams recorded on independent clocks
onto a single canonical, uniform 10 Hz time grid without extrapolation.

Interpolation Policy Table:
-----------------------------------------------------------------------------------------
Channel Name                  Source  Type         Interpolation Method
-----------------------------------------------------------------------------------------
timestamp_s                   Shared  float64      Common uniform 10 Hz grid (t=0 at start)
can_timestamp_s               CAN     float64      Linear interpolation
can_latitude_deg              CAN     float64      Linear interpolation
can_longitude_deg             CAN     float64      Linear interpolation
can_gps_velocity_kmh          CAN     float32      Linear interpolation
can_gps_heading_deg           CAN     float32      Circular (unwrapped radians) interpolation
can_height_m                  CAN     float32      Linear interpolation
can_gps_vertical_velocity_kmh CAN     float32      Linear interpolation
can_sample_period_s           CAN     float32      Linear interpolation
can_steering_angle_deg        CAN     float32      Linear interpolation
can_wheel_speed_fl_rad_s      CAN     float32      Linear interpolation
can_wheel_speed_fr_rad_s      CAN     float32      Linear interpolation
can_wheel_speed_rl_rad_s      CAN     float32      Linear interpolation
can_wheel_speed_rr_rad_s      CAN     float32      Linear interpolation
can_yaw_rate_deg_s            CAN     float32      Linear interpolation
can_indicated_vehicle_speed_kmh CAN   float32      Linear interpolation
can_indicated_longitudinal_accel_g CAN float32    Linear interpolation
can_indicated_lateral_accel_g CAN     float32      Linear interpolation
can_engine_speed_rpm          CAN     float32      Linear interpolation
can_coolant_temp_c            CAN     float32      Linear interpolation
can_brake_pressure_psi        CAN     float32      Linear interpolation
can_battery_voltage_v         CAN     float32      Linear interpolation
can_air_temp_c                CAN     float32      Linear interpolation
can_accelerator_pedal_pct     CAN     float32      Linear interpolation
can_gps_num_satellites        CAN     int32        Step (forward-fill, non-interpolated)
can_handbrake                 CAN     int8         Step (forward-fill, non-interpolated)
can_gear_requested            CAN     int8         Step (forward-fill, non-interpolated)
can_gear_actual               CAN     int8         Step (forward-fill, non-interpolated)
can_clutch_position           CAN     int8         Step (forward-fill, non-interpolated)
can_brake_position            CAN     int8         Step (forward-fill, non-interpolated)
phone_timestamp_ms            Phone   int64        Step (forward-fill, non-interpolated)
phone_timestamp_s             Phone   float64      Linear interpolation
phone_date_utc                Phone   string       Step (forward-fill, non-interpolated)
phone_latitude_deg            Phone   float64      Linear interpolation
phone_longitude_deg           Phone   float64      Linear interpolation
phone_altitude_m              Phone   float32      Linear interpolation
phone_gps_speed_raw           Phone   float32      Linear interpolation
phone_gps_speed_ms            Phone   float32      Linear interpolation
phone_gps_speed_kmh           Phone   float32      Linear interpolation
phone_gps_accuracy_m          Phone   float32      Linear interpolation
phone_gps_bearing_deg         Phone   float32      Circular (unwrapped radians) interpolation
phone_gps_satellites_in_range Phone   Int16        Step (forward-fill, non-interpolated)
phone_accel_x_m_s2            Phone   float32      Linear interpolation
phone_accel_y_m_s2            Phone   float32      Linear interpolation
phone_accel_z_m_s2            Phone   float32      Linear interpolation
phone_gravity_x_m_s2          Phone   float32      Linear interpolation
phone_gravity_y_m_s2          Phone   float32      Linear interpolation
phone_gravity_z_m_s2          Phone   float32      Linear interpolation
phone_gyro_x_rad_s            Phone   float32      Linear interpolation
phone_gyro_y_rad_s            Phone   float32      Linear interpolation
phone_gyro_z_rad_s            Phone   float32      Linear interpolation
phone_mag_x_uT                Phone   float32      Linear interpolation (preserves nulls)
phone_mag_y_uT                Phone   float32      Linear interpolation (preserves nulls)
phone_mag_z_uT                Phone   float32      Linear interpolation (preserves nulls)
phone_orientation_azimuth_deg Phone   float32      Circular (unwrapped radians) interpolation
phone_orientation_pitch_deg   Phone   float32      Linear interpolation
phone_orientation_roll_deg    Phone   float32      Linear interpolation
-----------------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal


# Categorical, count, or discrete flag columns that must NOT be linearly interpolated
CAN_STEP_COLUMNS = {
    "gps_num_satellites",
    "handbrake",
    "gear_requested",
    "gear_actual",
    "clutch_position",
    "brake_position",
}

PHONE_STEP_COLUMNS = {
    "gps_satellites_in_range",
    "date_utc",
    "timestamp_ms",
}

# Angular columns requiring circular unwrapping before interpolation
CAN_CIRCULAR_COLUMNS = {
    "gps_heading_deg",
}

PHONE_CIRCULAR_COLUMNS = {
    "gps_bearing_deg",
    "orientation_azimuth_deg",
}


@dataclass
class OffsetResult:
    """Estimated time offset and confidence metric between CAN and Phone streams."""

    estimated_offset_s: float
    offset_confidence: float
    method: str
    quality_flags: List[str] = field(default_factory=list)


@dataclass
class AlignedPairResult:
    """Result of time-aligning and resampling a paired CAN/Smartphone drive."""

    aligned_df: pd.DataFrame
    aligned_n_rows: int
    aligned_duration_s: float
    overlap_pct_of_can: float
    overlap_pct_of_phone: float


def estimate_offset(
    can_df: pd.DataFrame,
    phone_df: pd.DataFrame,
    max_lag_s: float = 30.0,
) -> OffsetResult:
    """Estimate clock offset (in seconds) between CAN and Smartphone recordings.

    Positive offset means the phone stream is delayed relative to CAN:
        t_phone = t_can + offset_s
        t_can = t_phone - offset_s

    Args:
        can_df: Vehicle CAN DataFrame with gps_velocity_kmh.
        phone_df: Smartphone DataFrame with gps_speed_kmh.
        max_lag_s: Maximum search window for time lag in seconds (+/- max_lag_s).

    Returns:
        OffsetResult containing estimated offset in seconds, peak correlation confidence, and quality flags.
    """
    flags: List[str] = []

    # Time axes at nominal 10 Hz rate
    t_c = np.arange(len(can_df)) * 0.1
    t_p = np.arange(len(phone_df)) * 0.1

    v_c = np.nan_to_num(can_df["gps_velocity_kmh"].values, nan=0.0)
    v_p = np.nan_to_num(phone_df["gps_speed_kmh"].values, nan=0.0)

    std_c = float(np.std(v_c))
    std_p = float(np.std(v_p))

    # Handle stationary runs (e.g. Vw1, Vw15 where car was parked the whole time)
    if std_c < 0.1 or std_p < 0.1:
        flags.append("stationary_run")
        return OffsetResult(
            estimated_offset_s=0.0,
            offset_confidence=0.0,
            method="stationary_fallback",
            quality_flags=flags,
        )

    # Resample onto a common 10 Hz grid across overlapping duration
    common_duration = min(t_c[-1], t_p[-1])
    if common_duration <= 1.0:
        flags.append("insufficient_duration")
        return OffsetResult(
            estimated_offset_s=0.0,
            offset_confidence=0.0,
            method="insufficient_duration",
            quality_flags=flags,
        )

    grid_t = np.arange(0.0, common_duration, 0.1)
    grid_c = np.interp(grid_t, t_c, v_c)
    grid_p = np.interp(grid_t, t_p, v_p)

    std_gc = float(np.std(grid_c))
    std_gp = float(np.std(grid_p))
    if std_gc < 0.1 or std_gp < 0.1:
        flags.append("stationary_run")
        return OffsetResult(
            estimated_offset_s=0.0,
            offset_confidence=0.0,
            method="stationary_fallback",
            quality_flags=flags,
        )

    # Normalize signals for Pearson correlation via cross-correlation
    c_norm = (grid_c - np.mean(grid_c)) / (std_gc * len(grid_c))
    p_norm = (grid_p - np.mean(grid_p)) / std_gp

    corr = signal.correlate(c_norm, p_norm, mode="full")
    lags = signal.correlation_lags(len(c_norm), len(p_norm), mode="full")

    # Restrict search window to [-max_lag_s, +max_lag_s]
    mask = np.abs(lags * 0.1) <= max_lag_s
    lags_w = lags[mask]
    corr_w = corr[mask]

    best_idx = int(np.argmax(corr_w))
    best_lag_s = -float(lags_w[best_idx] * 0.1)
    best_r = float(corr_w[best_idx])

    if best_r < 0.70:
        flags.append("low_confidence_offset")

    return OffsetResult(
        estimated_offset_s=best_lag_s,
        offset_confidence=best_r,
        method="gps_speed_cross_correlation",
        quality_flags=flags,
    )


def _step_interp(orig_t: np.ndarray, orig_vals: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Forward-fill step interpolation for discrete and categorical channels."""
    idx = np.searchsorted(orig_t, target_t, side="right") - 1
    idx = np.clip(idx, 0, len(orig_t) - 1)
    return orig_vals[idx]


def _circular_interp(orig_t: np.ndarray, orig_angles_deg: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Circular interpolation for angles in degrees [0, 360) via unwrapped radians."""
    if np.isnan(orig_angles_deg).all():
        return np.full_like(target_t, np.nan, dtype=np.float32)
    # Fill nan before unwrapping if sparse
    valid = ~np.isnan(orig_angles_deg)
    if valid.sum() < 2:
        return np.full_like(target_t, np.nan, dtype=np.float32)

    rads = np.unwrap(np.radians(orig_angles_deg[valid]))
    interp_rads = np.interp(target_t, orig_t[valid], rads)
    return (np.degrees(interp_rads) % 360.0).astype(np.float32)


def align_pair(
    can_df: pd.DataFrame,
    phone_df: pd.DataFrame,
    offset_s: float,
) -> AlignedPairResult:
    """Resample both CAN and Smartphone streams onto a shared uniform 10 Hz time grid.

    The common grid is strictly the intersection of available temporal coverage
    after applying offset_s, completely preventing extrapolation.

    Args:
        can_df: Vehicle CAN DataFrame.
        phone_df: Smartphone DataFrame.
        offset_s: Time offset between phone and CAN clocks in seconds (t_can = t_phone - offset_s).

    Returns:
        AlignedPairResult containing unified DataFrame, row counts, duration, and overlap percentages.
    """
    n_can = len(can_df)
    n_phone = len(phone_df)

    if n_can == 0 or n_phone == 0:
        raise ValueError("Cannot align empty DataFrame")

    t_can = np.arange(n_can) * 0.1
    t_phone = np.arange(n_phone) * 0.1

    # Phone time expressed on CAN clock: t_phone_on_can = t_phone - offset_s
    t_p_start_on_can = t_phone[0] - offset_s
    t_p_end_on_can = t_phone[-1] - offset_s

    t_c_start = t_can[0]
    t_c_end = t_can[-1]

    # Intersection window on CAN clock
    t_start = max(t_c_start, t_p_start_on_can)
    t_end = min(t_c_end, t_p_end_on_can)

    if t_start >= t_end:
        raise ValueError(
            f"No overlapping time range after applying offset {offset_s:+.3f}s: "
            f"CAN=[{t_c_start:.1f}, {t_c_end:.1f}], Phone_on_CAN=[{t_p_start_on_can:.1f}, {t_p_end_on_can:.1f}]"
        )

    # Build shared 10 Hz uniform time grid
    grid_t = np.arange(t_start, t_end + 1e-6, 0.1)
    # Strictly clip to avoid any floating-point endpoint overshoot (no extrapolation)
    grid_t = grid_t[(grid_t >= t_start - 1e-9) & (grid_t <= t_end + 1e-9)]
    n_aligned = len(grid_t)

    if n_aligned == 0:
        raise ValueError("Aligned intersection grid has zero samples")

    aligned_dict: Dict[str, np.ndarray] = {}

    # 1. Common shared time starting at 0.0s for the paired drive
    aligned_dict["timestamp_s"] = np.round(grid_t - grid_t[0], 4)

    # 2. Resample CAN channels
    for col in can_df.columns:
        out_col = f"can_{col}"
        series = can_df[col]
        vals = series.values

        if col in CAN_STEP_COLUMNS:
            aligned_dict[out_col] = _step_interp(t_can, vals, grid_t)
        elif col in CAN_CIRCULAR_COLUMNS:
            aligned_dict[out_col] = _circular_interp(t_can, vals.astype(np.float64), grid_t)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            interp_vals = np.interp(grid_t, t_can, vals.astype(np.float64))
            aligned_dict[out_col] = interp_vals.astype(series.dtype)
        else:
            aligned_dict[out_col] = _step_interp(t_can, vals, grid_t)

    # 3. Resample Smartphone channels onto (grid_t + offset_s)
    t_target_phone = grid_t + offset_s
    for col in phone_df.columns:
        out_col = f"phone_{col}"
        series = phone_df[col]
        vals = series.values

        if col in PHONE_STEP_COLUMNS:
            aligned_dict[out_col] = _step_interp(t_phone, vals, t_target_phone)
        elif col in PHONE_CIRCULAR_COLUMNS:
            aligned_dict[out_col] = _circular_interp(t_phone, vals.astype(np.float64), t_target_phone)
        elif series.isna().all():
            # Retain genuine nulls (e.g. missing magnetometer in French runs)
            aligned_dict[out_col] = np.full(n_aligned, np.nan, dtype=np.float32)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            interp_vals = np.interp(t_target_phone, t_phone, vals.astype(np.float64))
            aligned_dict[out_col] = interp_vals.astype(series.dtype)
        else:
            aligned_dict[out_col] = _step_interp(t_phone, vals, t_target_phone)

    aligned_df = pd.DataFrame(aligned_dict)

    # Enforce exact nullable Int16 for phone_gps_satellites_in_range
    if "phone_gps_satellites_in_range" in aligned_df.columns:
        aligned_df["phone_gps_satellites_in_range"] = pd.Series(
            aligned_dict["phone_gps_satellites_in_range"], dtype="Int16"
        )

    # Compute overlap metrics
    can_duration_s = max(t_can[-1] - t_can[0], 0.1)
    phone_duration_s = max(t_phone[-1] - t_phone[0], 0.1)
    aligned_duration_s = float(aligned_dict["timestamp_s"][-1])

    overlap_can_pct = min(round((aligned_duration_s / can_duration_s) * 100.0, 2), 100.0)
    overlap_phone_pct = min(round((aligned_duration_s / phone_duration_s) * 100.0, 2), 100.0)

    return AlignedPairResult(
        aligned_df=aligned_df,
        aligned_n_rows=n_aligned,
        aligned_duration_s=aligned_duration_s,
        overlap_pct_of_can=overlap_can_pct,
        overlap_pct_of_phone=overlap_phone_pct,
    )


def write_paired_parquet(df: pd.DataFrame, pair_id: str, out_dir: Path) -> Path:
    """Serialize aligned DataFrame into a deterministic Snappy-compressed Parquet file.

    Args:
        df: Aligned 57-column DataFrame.
        pair_id: Standardized pair identifier (e.g. 'pair_S1').
        out_dir: Destination directory.

    Returns:
        Path to the written Parquet file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair_id}.parquet"

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    return out_path
