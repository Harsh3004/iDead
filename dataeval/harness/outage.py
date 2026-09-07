"""GNSS Outage Simulator harness for IO-VNBD evaluation.

This module provides deterministic selection of outage windows, detection and
avoidance of authentic sensor gaps, null-masking of GNSS channels without data leakage,
and cross-stream ground truth trajectory extraction for dead-reckoning benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Standard benchmark outage durations in seconds
DEFAULT_OUTAGE_DURATIONS: List[float] = [10.0, 30.0, 60.0, 120.0, 180.0]

# Standard edge margin in seconds (leading and trailing buffer)
# 10.0s corresponds to 100 samples at 10 Hz, giving adequate time for EKF bias
# convergence (accel/gyro biases, yaw lock) before outage onset, and adequate time
# after outage to observe re-acquisition and compute terminal drift without truncating.
DEFAULT_EDGE_MARGIN_S: float = 10.0

# Canonical GNSS columns to blank per data source
CAN_GNSS_COLUMNS: List[str] = [
    "latitude_deg",
    "longitude_deg",
    "gps_velocity_kmh",
    "gps_heading_deg",
    "gps_vertical_velocity_kmh",
    "gps_num_satellites",
]

CAN_GNSS_ADJACENT_COLUMNS: List[str] = [
    "height_m",  # VBOX GPS ellipsoidal/orthometric height
]

PHONE_GNSS_COLUMNS: List[str] = [
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
    "gps_speed_ms",
    "gps_speed_kmh",
    "gps_accuracy_m",
    "gps_bearing_deg",
    "gps_satellites_in_range",
]

PHONE_GNSS_ADJACENT_COLUMNS: List[str] = [
    "gps_speed_raw",  # Raw unscaled GPS speed channel
]


@dataclass
class OutageWindow:
    """Represents a discrete synthetic GNSS outage window."""

    run_id: str
    source_side: str  # "phone", "can", or "both"
    start_s: float
    duration_s: float
    end_s: float
    overlaps_real_gap: bool = False

    def __init__(
        self,
        run_id: Optional[str] = None,
        source_side: str = "phone",
        start_s: float = 0.0,
        duration_s: float = 0.0,
        end_s: Optional[float] = None,
        overlaps_real_gap: bool = False,
        pair_id: Optional[str] = None,
        run_or_pair_id: Optional[str] = None,
    ):
        self.run_id = run_id or pair_id or run_or_pair_id or ""
        self.source_side = source_side
        self.start_s = float(start_s)
        self.duration_s = float(duration_s)
        self.end_s = float(end_s if end_s is not None else (self.start_s + self.duration_s))
        self.overlaps_real_gap = bool(overlaps_real_gap)

    @property
    def pair_id(self) -> str:
        return self.run_id

    @property
    def run_or_pair_id(self) -> str:
        return self.run_id


@dataclass
class OutageSkip:
    """Record of an attempted (run, duration) outage that could not be placed."""

    run_id: str
    duration_s: float
    reason: str


class OutageWindowList(list):
    """Subclass of list holding OutageWindow objects with companion skipped metadata."""

    def __init__(
        self,
        windows: Optional[Sequence[OutageWindow]] = None,
        skipped: Optional[Sequence[OutageSkip]] = None,
    ):
        super().__init__(windows or [])
        self.skipped: List[OutageSkip] = list(skipped or [])

    @property
    def windows(self) -> OutageWindowList:
        return self


def find_real_gaps(
    df: pd.DataFrame,
    gap_threshold_s: float = 2.0,
    side: str = "phone",
) -> List[Tuple[float, float]]:
    """Detect authentic recording / timestamp gaps in a run DataFrame.

    Args:
        df: Input run DataFrame with 'timestamp_s' column.
        gap_threshold_s: Minimum inter-sample gap duration to qualify as real gap (default 2.0s).
        side: Target stream being checked ('phone', 'can', or 'both').

    Returns:
        Sorted list of (gap_start_s, gap_end_s) intervals in the primary timestamp_s reference frame.
    """
    gaps: List[Tuple[float, float]] = []

    if df.empty or "timestamp_s" not in df.columns or len(df) < 2:
        return gaps

    ts = df["timestamp_s"].values
    dt = np.diff(ts)
    gap_indices = np.where(dt > gap_threshold_s)[0]
    for idx in gap_indices:
        gaps.append((float(ts[idx]), float(ts[idx + 1])))

    # Check phone-specific jumps in paired data
    if side in ("phone", "both") and "phone_timestamp_s" in df.columns:
        pts = df["phone_timestamp_s"].values
        pdt = np.diff(pts)
        p_gap_indices = np.where(pdt > gap_threshold_s)[0]
        for idx in p_gap_indices:
            gaps.append((float(ts[idx]), float(ts[idx + 1])))

    # Check CAN-specific jumps in paired data
    if side in ("can", "both") and "can_timestamp_s" in df.columns:
        cts = df["can_timestamp_s"].values
        cdt = np.diff(cts)
        c_gap_indices = np.where(cdt > gap_threshold_s)[0]
        for idx in c_gap_indices:
            gaps.append((float(ts[idx]), float(ts[idx + 1])))

    if not gaps:
        return []

    # Merge overlapping or contiguous gap intervals
    gaps.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [gaps[0]]
    for cur_start, cur_end in gaps[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    return merged


def _subtract_interval(
    intervals: List[Tuple[float, float]],
    exclude_start: float,
    exclude_end: float,
) -> List[Tuple[float, float]]:
    """Subtract [exclude_start, exclude_end] from a list of disjoint intervals."""
    result: List[Tuple[float, float]] = []
    for a, b in intervals:
        if exclude_end <= a or exclude_start >= b:
            # No overlap
            result.append((a, b))
        else:
            # Overlap exists
            if a < exclude_start:
                result.append((a, exclude_start))
            if exclude_end < b:
                result.append((exclude_end, b))
    return [(round(a, 3), round(b, 3)) for a, b in result if round(b - a, 3) > 0.0]


def select_outage_windows(
    df: pd.DataFrame,
    run_id: str,
    durations: Sequence[float] = DEFAULT_OUTAGE_DURATIONS,
    side: str = "phone",
    n_per_duration: int = 1,
    seed: int = 42,
    avoid_real_gaps: bool = True,
    avoid_run_edges: bool = True,
    edge_margin_s: float = DEFAULT_EDGE_MARGIN_S,
    gap_threshold_s: float = 2.0,
    return_skipped: bool = False,
) -> Union[OutageWindowList, Tuple[OutageWindowList, List[OutageSkip]]]:
    """Deterministically choose candidate outage windows within usable run duration.

    Args:
        df: Input run DataFrame.
        run_id: Unique identifier for the run or pair.
        durations: List of outage durations in seconds (default [10, 30, 60, 120, 180]).
        side: Target stream to blank ('phone', 'can', or 'both').
        n_per_duration: Number of non-overlapping windows to select per duration (default 1).
        seed: Master random seed for deterministic generation.
        avoid_real_gaps: If True, avoids placing synthetic outages across real recording gaps.
        avoid_run_edges: If True, enforces edge_margin_s leading/trailing context around outage.
        edge_margin_s: Margin duration in seconds to keep before and after outage.
        gap_threshold_s: Minimum inter-sample gap duration to identify real gaps.
        return_skipped: If True, returns (OutageWindowList, List[OutageSkip]) tuple.

    Returns:
        OutageWindowList containing selected OutageWindow instances and .skipped attribute,
        or a tuple of (OutageWindowList, List[OutageSkip]) if return_skipped is True.
    """
    windows: List[OutageWindow] = []
    skipped: List[OutageSkip] = []

    if df.empty or "timestamp_s" not in df.columns:
        for d in durations:
            skipped.append(
                OutageSkip(
                    run_id=run_id,
                    duration_s=float(d),
                    reason="empty_or_invalid_dataframe: missing timestamp_s column or 0 rows",
                )
            )
        res = OutageWindowList(windows, skipped)
        return (res, skipped) if return_skipped else res

    min_ts = float(df["timestamp_s"].min())
    max_ts = float(df["timestamp_s"].max())
    total_span = max_ts - min_ts

    margin = float(edge_margin_s) if avoid_run_edges else 0.0
    real_gaps = find_real_gaps(df, gap_threshold_s=gap_threshold_s, side=side)

    for dur in durations:
        dur_f = float(dur)
        min_required_span = dur_f + 2.0 * margin

        if total_span < min_required_span:
            skipped.append(
                OutageSkip(
                    run_id=run_id,
                    duration_s=dur_f,
                    reason=(
                        f"run_too_short: run total span {total_span:.1f}s is less than "
                        f"required {dur_f:.1f}s + 2*{margin:.1f}s margin ({min_required_span:.1f}s)"
                    ),
                )
            )
            continue

        earliest_start = min_ts + margin
        latest_start = max_ts - margin - dur_f

        if latest_start < earliest_start:
            skipped.append(
                OutageSkip(
                    run_id=run_id,
                    duration_s=dur_f,
                    reason=f"run_too_short: latest start {latest_start:.1f}s < earliest start {earliest_start:.1f}s",
                )
            )
            continue

        # Initial valid start-time domain
        valid_intervals: List[Tuple[float, float]] = [(earliest_start, latest_start)]

        if avoid_real_gaps and real_gaps:
            for g_start, g_end in real_gaps:
                conflict_start = g_start - dur_f
                conflict_end = g_end
                valid_intervals = _subtract_interval(valid_intervals, conflict_start, conflict_end)

        total_valid_len = sum(b - a for a, b in valid_intervals)
        if total_valid_len <= 0.0:
            skipped.append(
                OutageSkip(
                    run_id=run_id,
                    duration_s=dur_f,
                    reason=(
                        f"overlaps_real_gap: no gap-free window of duration {dur_f:.1f}s fits in run"
                        if avoid_real_gaps
                        else "no_valid_interval: start time domain is empty"
                    ),
                )
            )
            continue

        # Select n_per_duration windows
        cur_intervals = list(valid_intervals)
        windows_for_dur: List[OutageWindow] = []

        for idx in range(n_per_duration):
            cur_len = sum(b - a for a, b in cur_intervals if b > a)
            if cur_len <= 0.0:
                skipped.append(
                    OutageSkip(
                        run_id=run_id,
                        duration_s=dur_f,
                        reason=(
                            f"no_non_overlapping_slot: could only place {len(windows_for_dur)} "
                            f"of {n_per_duration} requested windows for duration {dur_f:.1f}s"
                        ),
                    )
                )
                break

            # Deterministic, reproducible RNG derivation per (run, side, duration, index)
            hasher = hashlib.sha256(f"{seed}_{run_id}_{side}_{dur_f}_{idx}".encode("utf-8"))
            inst_seed = int(hasher.hexdigest()[:8], 16)
            rng = np.random.default_rng(inst_seed)

            u = float(rng.uniform(0.0, cur_len))
            accum = 0.0
            chosen_start: Optional[float] = None
            for a, b in cur_intervals:
                seg_len = b - a
                if seg_len <= 0:
                    continue
                if accum + seg_len >= u:
                    offset = u - accum
                    chosen_start = a + offset
                    break
                accum += seg_len

            if chosen_start is None:
                chosen_start = cur_intervals[0][0]

            # Snap to 0.1s grid (10 Hz aligned)
            chosen_start = round(chosen_start, 1)
            # Clamp within bounds
            chosen_start = max(earliest_start, min(latest_start, chosen_start))
            chosen_end = round(chosen_start + dur_f, 1)

            # Check if this window overlaps any real recording gap
            overlaps = any(
                max(chosen_start, g_start) < min(chosen_end, g_end)
                for g_start, g_end in real_gaps
            )

            win = OutageWindow(
                run_id=run_id,
                source_side=side,
                start_s=chosen_start,
                duration_s=dur_f,
                end_s=chosen_end,
                overlaps_real_gap=overlaps,
            )
            windows_for_dur.append(win)
            windows.append(win)

            # If selecting more windows for this duration, remove [chosen_start - dur_f, chosen_end]
            if n_per_duration > 1:
                cur_intervals = _subtract_interval(
                    cur_intervals, chosen_start - dur_f, chosen_end
                )

    result_list = OutageWindowList(windows, skipped)
    if return_skipped:
        return result_list, skipped
    return result_list


def apply_outage(df: pd.DataFrame, window: OutageWindow) -> pd.DataFrame:
    """Apply a synthetic GNSS outage window to a run DataFrame.

    Sets all GNSS channels for the specified source_side to genuine null/NaN
    for rows in [window.start_s, window.end_s). Adds boolean column 'gnss_outage_active'.
    All IMU, wheel speed, steering, and chassis odometry channels are provably untouched.

    Args:
        df: Input run DataFrame.
        window: OutageWindow defining start_s, end_s, and source_side.

    Returns:
        Copy of df with GNSS channels masked to null and 'gnss_outage_active' flag added.
    """
    out_df = df.copy()

    if "timestamp_s" not in out_df.columns or out_df.empty:
        out_df["gnss_outage_active"] = False
        return out_df

    # Outage mask: half-open interval [start_s, end_s)
    active_mask = (out_df["timestamp_s"] >= window.start_s) & (
        out_df["timestamp_s"] < window.end_s
    )
    out_df["gnss_outage_active"] = active_mask

    # Determine target columns to blank
    cols_to_blank: List[str] = []

    if window.source_side in ("can", "both"):
        can_targets = CAN_GNSS_COLUMNS + CAN_GNSS_ADJACENT_COLUMNS
        for col in can_targets:
            prefixed = f"can_{col}"
            if prefixed in out_df.columns:
                cols_to_blank.append(prefixed)
            if col in out_df.columns:
                cols_to_blank.append(col)

    if window.source_side in ("phone", "both"):
        phone_targets = PHONE_GNSS_COLUMNS + PHONE_GNSS_ADJACENT_COLUMNS
        for col in phone_targets:
            prefixed = f"phone_{col}"
            if prefixed in out_df.columns:
                cols_to_blank.append(prefixed)
            if col in out_df.columns:
                cols_to_blank.append(col)

    # Blank columns inside active_mask with real nulls
    for col in cols_to_blank:
        if col not in out_df.columns:
            continue

        dtype = out_df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            if not isinstance(dtype, pd.Int64Dtype) and not isinstance(dtype, pd.Int32Dtype) and not isinstance(dtype, pd.Int16Dtype):
                out_df[col] = out_df[col].astype("Int32")
            out_df.loc[active_mask, col] = pd.NA
        elif pd.api.types.is_float_dtype(dtype):
            out_df.loc[active_mask, col] = np.nan
        else:
            out_df.loc[active_mask, col] = None

    return out_df


def ground_truth_trajectory(
    df: pd.DataFrame,
    window: OutageWindow,
) -> Optional[pd.DataFrame]:
    """Extract untouched real GPS trajectory for the masked window from the other stream.

    For paired runs, when one stream is blanked (e.g. phone), the unblanked counterpart
    (e.g. CAN) serves as independent displacement ground truth during the outage.
    For unpaired runs, returns None with documented limitation.

    Args:
        df: Input run DataFrame (paired or unpaired, raw or masked).
        window: OutageWindow defining start_s, end_s, and source_side.

    Returns:
        DataFrame with canonical trajectory columns ('timestamp_s', 'latitude_deg',
        'longitude_deg', 'altitude_m', 'speed_ms', 'heading_deg') for rows in
        [window.start_s, window.end_s), or None if cross-stream ground truth is unavailable.
    """
    if df.empty or "timestamp_s" not in df.columns:
        return None

    if window.source_side == "both":
        return None

    mask = (df["timestamp_s"] >= window.start_s) & (df["timestamp_s"] < window.end_s)
    sub = df[mask]
    if sub.empty:
        return None

    if window.source_side == "phone":
        if "can_latitude_deg" not in df.columns or "can_longitude_deg" not in df.columns:
            return None

        gt = pd.DataFrame(index=sub.index)
        gt["timestamp_s"] = sub["timestamp_s"].values
        gt["latitude_deg"] = sub["can_latitude_deg"].values
        gt["longitude_deg"] = sub["can_longitude_deg"].values
        gt["altitude_m"] = (
            sub["can_height_m"].values
            if "can_height_m" in sub.columns
            else np.full(len(sub), np.nan, dtype=np.float32)
        )
        gt["speed_ms"] = (
            (sub["can_gps_velocity_kmh"].values / 3.6).astype(np.float32)
            if "can_gps_velocity_kmh" in sub.columns
            else np.full(len(sub), np.nan, dtype=np.float32)
        )
        gt["heading_deg"] = (
            sub["can_gps_heading_deg"].values
            if "can_gps_heading_deg" in sub.columns
            else np.full(len(sub), np.nan, dtype=np.float32)
        )
        return gt

    elif window.source_side == "can":
        if "phone_latitude_deg" not in df.columns or "phone_longitude_deg" not in df.columns:
            return None

        gt = pd.DataFrame(index=sub.index)
        gt["timestamp_s"] = sub["timestamp_s"].values
        gt["latitude_deg"] = sub["phone_latitude_deg"].values
        gt["longitude_deg"] = sub["phone_longitude_deg"].values
        gt["altitude_m"] = (
            sub["phone_altitude_m"].values
            if "phone_altitude_m" in sub.columns
            else np.full(len(sub), np.nan, dtype=np.float32)
        )
        gt["speed_ms"] = (
            sub["phone_gps_speed_ms"].values
            if "phone_gps_speed_ms" in sub.columns
            else np.full(len(sub), np.nan, dtype=np.float32)
        )
        gt["heading_deg"] = (
            sub["phone_gps_bearing_deg"].values
            if "phone_gps_bearing_deg" in sub.columns
            else np.full(len(sub), np.nan, dtype=np.float32)
        )
        return gt

    return None
