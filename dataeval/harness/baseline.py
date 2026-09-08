"""Naive dead-reckoning baseline: constant-velocity and constant-heading propagation.

This module provides the baseline navigation floor for evaluating GNSS outages.
It estimates the initial state solely from pre-outage GNSS measurements and propagates
position forward along a great circle under constant speed and heading assumptions.

NOTE: This baseline intentionally does NOT perform IMU integration, wheel speed odometry,
or sensor fusion. It represents the 'what if you did nothing clever at all' baseline floor
against which all subsequent inertial fusion engines (Steps 9-11) and machine learning
models (Module C) are benchmarked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import pandas as pd

from dataeval.harness.outage import OutageWindow


# Standard pre-outage averaging window in seconds.
# 1.0s corresponds to 10 samples at 10 Hz, sufficiently attenuating GPS Doppler noise
# without introducing phase lag during vehicle maneuvers (turns, braking).
DEFAULT_PRE_OUTAGE_WINDOW_S: float = 1.0

# Mean Earth radius in meters (WGS-84 spherical approximation)
EARTH_RADIUS_M: float = 6371000.0


@dataclass
class PreOutageState:
    """Estimated dynamic state immediately preceding a GNSS outage."""

    t0: float
    lat0: float
    lon0: float
    v0: float  # Speed in m/s
    theta0: float  # Heading in degrees clockwise from true North [0.0, 360.0)


def extract_pre_outage_state(
    df: pd.DataFrame,
    window: OutageWindow,
    pre_outage_window_s: float = DEFAULT_PRE_OUTAGE_WINDOW_S,
) -> Optional[PreOutageState]:
    """Extract initial position, speed, and heading strictly before outage onset.

    Crucially, this function strictly inspects data rows where timestamp_s < window.start_s,
    ensuring zero information leakage from inside or after the outage window.

    Args:
        df: Outage-masked or original run DataFrame.
        window: OutageWindow defining start_s and target source_side.
        pre_outage_window_s: Lookback duration in seconds to average speed and heading.

    Returns:
        PreOutageState instance, or None if insufficient pre-outage GNSS fixes exist.
    """
    if df.empty or "timestamp_s" not in df.columns:
        return None

    # Strictly pre-outage data: t < window.start_s
    pre_mask = df["timestamp_s"] < window.start_s
    pre_df = df[pre_mask]
    if pre_df.empty:
        return None

    # Identify source channels based on source_side and schema
    side = window.source_side.lower()

    if side in ("phone", "both") and "phone_latitude_deg" in pre_df.columns:
        # Paired dataset - phone target
        lat_col = "phone_latitude_deg"
        lon_col = "phone_longitude_deg"
        spd_col = "phone_gps_speed_ms"
        brg_col = "phone_gps_bearing_deg"
        is_kmh = False
    elif side in ("can", "both") and "can_latitude_deg" in pre_df.columns:
        # Paired dataset - can target
        lat_col = "can_latitude_deg"
        lon_col = "can_longitude_deg"
        spd_col = "can_gps_velocity_kmh"
        brg_col = "can_gps_heading_deg"
        is_kmh = True
    elif "latitude_deg" in pre_df.columns:
        # Unpaired single-stream dataset
        lat_col = "latitude_deg"
        lon_col = "longitude_deg"
        if "gps_speed_ms" in pre_df.columns:
            spd_col = "gps_speed_ms"
            is_kmh = False
        elif "gps_velocity_kmh" in pre_df.columns:
            spd_col = "gps_velocity_kmh"
            is_kmh = True
        else:
            spd_col = None
        brg_col = "gps_bearing_deg" if "gps_bearing_deg" in pre_df.columns else "gps_heading_deg"
        if brg_col not in pre_df.columns:
            brg_col = None
    else:
        return None

    # Filter to non-null position fixes
    valid_pre = pre_df.dropna(subset=[lat_col, lon_col])
    if valid_pre.empty:
        return None

    # Window slice: [start_s - pre_outage_window_s, start_s)
    window_slice = valid_pre[valid_pre["timestamp_s"] >= (window.start_s - pre_outage_window_s)]
    # Fallback to last available fixes if window_slice has fewer than 2 points (e.g. 2 Hz throttled cohort)
    if len(window_slice) < 2:
        window_slice = valid_pre.tail(max(2, min(len(valid_pre), 10)))

    last_fix = window_slice.iloc[-1]
    t0 = float(last_fix["timestamp_s"])
    lat0 = float(last_fix[lat_col])
    lon0 = float(last_fix[lon_col])

    # Speed estimation (mean across slice)
    if spd_col and spd_col in window_slice.columns:
        valid_spds = window_slice[spd_col].dropna()
        if not valid_spds.empty:
            raw_v = float(valid_spds.mean())
            v0 = (raw_v / 3.6) if is_kmh else raw_v
        else:
            v0 = 0.0
    else:
        v0 = 0.0

    # Heading estimation (circular mean across slice)
    if brg_col and brg_col in window_slice.columns:
        valid_brgs = window_slice[brg_col].dropna().values
        if len(valid_brgs) > 0:
            rads = np.deg2rad(valid_brgs)
            sin_mean = float(np.mean(np.sin(rads)))
            cos_mean = float(np.mean(np.cos(rads)))
            theta0 = float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)
        else:
            theta0 = 0.0
    else:
        theta0 = 0.0

    return PreOutageState(t0=t0, lat0=lat0, lon0=lon0, v0=v0, theta0=theta0)


def propagate_constant_velocity_heading(
    pre_state: PreOutageState,
    target_timestamps: Sequence[float],
) -> pd.DataFrame:
    """Propagate position forward at constant speed and heading along a great circle.

    Uses the direct geodetic problem on a spherical Earth of radius R = 6,371,000 m.

    Args:
        pre_state: PreOutageState containing initial (t0, lat0, lon0, v0, theta0).
        target_timestamps: Sequence of timestamps to generate predictions for.

    Returns:
        DataFrame with columns 'timestamp_s', 'latitude_deg', 'longitude_deg',
        'speed_ms', 'heading_deg'.
    """
    ts = np.asarray(target_timestamps, dtype=np.float64)
    n = len(ts)
    if n == 0:
        return pd.DataFrame(
            columns=["timestamp_s", "latitude_deg", "longitude_deg", "speed_ms", "heading_deg"]
        )

    dt = ts - pre_state.t0
    dist = pre_state.v0 * dt  # meters

    phi0 = np.deg2rad(pre_state.lat0)
    lam0 = np.deg2rad(pre_state.lon0)
    brg = np.deg2rad(pre_state.theta0)

    d_R = dist / EARTH_RADIUS_M

    # Great circle direct geodetic formula
    phi_pred = np.arcsin(
        np.sin(phi0) * np.cos(d_R) + np.cos(phi0) * np.sin(d_R) * np.cos(brg)
    )
    lam_pred = lam0 + np.arctan2(
        np.sin(brg) * np.sin(d_R) * np.cos(phi0),
        np.cos(d_R) - np.sin(phi0) * np.sin(phi_pred),
    )

    pred_lats = np.rad2deg(phi_pred)
    # Normalize longitude to [-180, 180)
    pred_lons = (np.rad2deg(lam_pred) + 180.0) % 360.0 - 180.0

    return pd.DataFrame({
        "timestamp_s": ts,
        "latitude_deg": pred_lats,
        "longitude_deg": pred_lons,
        "speed_ms": np.full(n, pre_state.v0, dtype=np.float32),
        "heading_deg": np.full(n, pre_state.theta0, dtype=np.float32),
    })
