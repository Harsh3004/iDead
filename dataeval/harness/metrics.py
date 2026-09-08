"""Evaluation metrics engine for dead-reckoning GNSS outage benchmarking.

Computes the seven locked leaderboard metrics against independent ground truth trajectories:
- final_pos_error_m (haversine endpoint drift)
- pct_of_distance (final error as percentage of true path distance; NaN for stationary runs)
- cep50_m / cep95_m (50th and 95th percentile error across entire outage window)
- along_track_m / cross_track_m (projection parallel and perpendicular to travel direction)
- heading_error_deg (endpoint heading difference wrapped into [0, 180] deg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


EARTH_RADIUS_M: float = 6371000.0
MIN_TRAVEL_DISTANCE_M: float = 1.0


@dataclass
class OutageMetrics:
    """Computed leaderboard metrics for an outage instance."""

    final_pos_error_m: float = np.nan
    pct_of_distance: float = np.nan
    cep50_m: float = np.nan
    cep95_m: float = np.nan
    along_track_m: float = np.nan
    cross_track_m: float = np.nan
    heading_error_deg: float = np.nan
    true_distance_m: float = np.nan
    status: str = "ok"
    skip_reason: str = ""


def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute great-circle haversine distance between coordinates on spherical Earth in meters."""
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)

    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_M * c


def wrap_heading_error(
    pred_heading_deg: float,
    gt_heading_deg: float,
) -> float:
    """Compute shortest angular difference between two headings in degrees, wrapped to [0.0, 180.0]."""
    if pd.isna(pred_heading_deg) or pd.isna(gt_heading_deg):
        return np.nan

    diff = (pred_heading_deg - gt_heading_deg + 180.0) % 360.0 - 180.0
    return float(abs(diff))


def decompose_along_cross_track(
    pred_lat: float,
    pred_lon: float,
    gt_lat: float,
    gt_lon: float,
    track_heading_deg: float,
) -> Tuple[float, float]:
    """Decompose position error into along-track and cross-track components in local ENU frame.

    Args:
        pred_lat, pred_lon: Predicted endpoint position in degrees.
        gt_lat, gt_lon: True endpoint position in degrees.
        track_heading_deg: True direction of travel in degrees clockwise from North.

    Returns:
        tuple (along_track_m, cross_track_m):
        - along_track_m: Displacement parallel to heading (positive = ahead/overshooting).
        - cross_track_m: Displacement perpendicular to heading (positive = to the right of track).
    """
    if (
        pd.isna(pred_lat)
        or pd.isna(pred_lon)
        or pd.isna(gt_lat)
        or pd.isna(gt_lon)
        or pd.isna(track_heading_deg)
    ):
        return np.nan, np.nan

    # Local East-North tangent plane displacement in meters
    e_E = (pred_lon - gt_lon) * (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(gt_lat))
    e_N = (pred_lat - gt_lat) * (np.pi / 180.0) * EARTH_RADIUS_M

    psi = np.deg2rad(track_heading_deg)
    along_track = e_E * np.sin(psi) + e_N * np.cos(psi)
    cross_track = e_E * np.cos(psi) - e_N * np.sin(psi)

    return float(along_track), float(cross_track)


def compute_outage_metrics(
    pred_df: pd.DataFrame,
    gt_df: Optional[pd.DataFrame],
    min_travel_dist_m: float = MIN_TRAVEL_DISTANCE_M,
) -> OutageMetrics:
    """Compute all locked leaderboard metrics comparing predicted trajectory against ground truth.

    Args:
        pred_df: Predicted trajectory with 'timestamp_s', 'latitude_deg', 'longitude_deg',
                 and optional 'heading_deg'.
        gt_df: Ground-truth trajectory with matching timestamps, 'latitude_deg', 'longitude_deg',
               and optional 'heading_deg', or None if cross-stream GT is unavailable.
        min_travel_dist_m: Minimum true distance required to compute pct_of_distance (default 1.0m).

    Returns:
        OutageMetrics dataclass populated with all metric values.
    """
    if gt_df is None or gt_df.empty or len(gt_df) < 2:
        return OutageMetrics(
            status="no_ground_truth",
            skip_reason="unpaired_run: no independent cross-stream ground truth",
        )

    if pred_df.empty or len(pred_df) < 2:
        return OutageMetrics(
            status="error",
            skip_reason="insufficient_predictions: predicted trajectory has < 2 points",
        )

    # Check for valid coordinate data in GT
    valid_gt = gt_df.dropna(subset=["latitude_deg", "longitude_deg"])
    if len(valid_gt) < 2 or (len(valid_gt) / len(gt_df)) < 0.5:
        return OutageMetrics(
            status="error",
            skip_reason="insufficient_ground_truth_coverage: GT trajectory contains excessive nulls",
        )

    # Align timestamps between pred and GT
    pred_aligned = pred_df.copy()
    if not np.array_equal(pred_aligned["timestamp_s"].values, gt_df["timestamp_s"].values):
        # Interpolate predictions onto GT timestamps if grid differs
        interp_lat = np.interp(
            gt_df["timestamp_s"].values,
            pred_aligned["timestamp_s"].values,
            pred_aligned["latitude_deg"].values,
        )
        interp_lon = np.interp(
            gt_df["timestamp_s"].values,
            pred_aligned["timestamp_s"].values,
            pred_aligned["longitude_deg"].values,
        )
        pred_lats = interp_lat
        pred_lons = interp_lon
        pred_headings = np.interp(
            gt_df["timestamp_s"].values,
            pred_aligned["timestamp_s"].values,
            pred_aligned["heading_deg"].values,
        )
    else:
        pred_lats = pred_aligned["latitude_deg"].values
        pred_lons = pred_aligned["longitude_deg"].values
        pred_headings = pred_aligned["heading_deg"].values

    gt_lats = gt_df["latitude_deg"].values
    gt_lons = gt_df["longitude_deg"].values

    # 1. Point-by-point position errors across entire outage window
    errors = haversine_distance(pred_lats, pred_lons, gt_lats, gt_lons)
    valid_errors = errors[~np.isnan(errors)]

    if len(valid_errors) == 0:
        return OutageMetrics(
            status="error",
            skip_reason="all_errors_nan: no valid coordinate comparisons",
        )

    final_pos_error_m = float(valid_errors[-1])
    cep50_m = float(np.percentile(valid_errors, 50))
    cep95_m = float(np.percentile(valid_errors, 95))

    # 2. True distance travelled along ground truth path
    seg_dists = haversine_distance(gt_lats[:-1], gt_lons[:-1], gt_lats[1:], gt_lons[1:])
    true_distance_m = float(np.sum(seg_dists[~np.isnan(seg_dists)]))

    # 3. Percentage of distance (handled cleanly for stationary runs)
    if true_distance_m >= min_travel_dist_m:
        pct_of_distance = float(final_pos_error_m / true_distance_m * 100.0)
        skip_reason = ""
    else:
        # Stationary run: distance is near-zero (GPS jitter)
        pct_of_distance = np.nan
        skip_reason = f"stationary_run: true distance {true_distance_m:.2f}m < {min_travel_dist_m:.1f}m threshold"

    # 4. Along-track and cross-track error at endpoint
    if "heading_deg" in gt_df.columns and not gt_df["heading_deg"].dropna().empty:
        valid_h = gt_df["heading_deg"].dropna()
        final_gt_heading = float(valid_h.iloc[-1])
    else:
        # Fallback to path chord bearing if heading channel absent
        dlat = np.deg2rad(gt_lats[-1] - gt_lats[0])
        dlon = np.deg2rad(gt_lons[-1] - gt_lons[0])
        p1 = np.deg2rad(gt_lats[0])
        p2 = np.deg2rad(gt_lats[-1])
        y = np.sin(dlon) * np.cos(p2)
        x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlon)
        final_gt_heading = float(np.rad2deg(np.arctan2(y, x)) % 360.0)

    along_track_m, cross_track_m = decompose_along_cross_track(
        pred_lat=pred_lats[-1],
        pred_lon=pred_lons[-1],
        gt_lat=gt_lats[-1],
        gt_lon=gt_lons[-1],
        track_heading_deg=final_gt_heading,
    )

    # 5. Heading error
    pred_final_heading = float(pred_headings[-1])
    heading_error_deg = wrap_heading_error(pred_final_heading, final_gt_heading)

    return OutageMetrics(
        final_pos_error_m=final_pos_error_m,
        pct_of_distance=pct_of_distance,
        cep50_m=cep50_m,
        cep95_m=cep95_m,
        along_track_m=along_track_m,
        cross_track_m=cross_track_m,
        heading_error_deg=heading_error_deg,
        true_distance_m=true_distance_m,
        status="ok",
        skip_reason=skip_reason,
    )
