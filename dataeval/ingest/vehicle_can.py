"""Vehicle/CAN dataset ingestion module for IO-VNBD.

This module provides discovery, deduplication, schema normalization, typed casting,
validation, and Parquet serialization for the 29-column vehicle/CAN CSV dataset.

Canonical Column Schema Definition:
------------------------------------------------------------------------------------------------------
Canonical Column Name             Dtype    Unit       Source Raw Header
------------------------------------------------------------------------------------------------------
gps_num_satellites                int32    count      No of GPS Satellites Available
timestamp_s                       float64  s          Time Since Start of Day (seconds)
latitude_deg                      float64  deg        Latitude (degrees)
longitude_deg                     float64  deg        Longitude (degrees)
gps_velocity_kmh                  float32  km/h       Velocity (km/hr)
gps_heading_deg                   float32  deg        Heading (degrees)
height_m                          float32  m          Height (km)  [CORRECTED: raw says km, values are m]
gps_vertical_velocity_kmh         float32  km/h       Vertical velocity (km/hr)
sample_period_s                   float32  s          Sample period (seconds)
steering_angle_deg                float32  deg        Steering Angle (degrees)
wheel_speed_fl_rad_s              float32  rad/s      Wheel Speed Front Left (rad/sec)
wheel_speed_fr_rad_s              float32  rad/s      Wheel Speed Front Right (rad/sec)
wheel_speed_rl_rad_s              float32  rad/s      Wheel Speed Rear Left (rad/sec)
wheel_speed_rr_rad_s              float32  rad/s      Wheel Speed Rear Right (rad/sec)
yaw_rate_deg_s                    float32  deg/s      Yaw Rate (deg/sec)
indicated_vehicle_speed_kmh       float32  km/h       Indicated Vehicle Speed (km/hr)
indicated_longitudinal_accel_g    float32  g          Indicated Longitudinal Acceleration (g)
indicated_lateral_accel_g         float32  g          Indicated Lateral Acceleration (g)
handbrake                         int8     flag (0/1) Handbrake (0 or 1)
gear_requested                    int8     gear code  Gear Requested (Number fof gear employed 1-5)
gear_actual                       int8     gear code  Gear (Number fof gear employed 1-5)
engine_speed_rpm                  float32  rpm        Engine Speed (rev/min)
coolant_temp_c                    float32  °C         Coolant Temperature (degrees)
clutch_position                   int8     flag (0/1) Clutch Position (0 or 1)  [NOTE: 100% zero/unconnected]
brake_pressure_psi                float32  psi        Brake Pressure (psi)
brake_position                    int8     flag (0/1) Brake Position (0 or 1)
battery_voltage_v                 float32  V          Battery Voltage (volts)
air_temp_c                        float32  °C         Air Temperature (degrees)
accelerator_pedal_pct             float32  %          Accelerator Pedal Position (0 or 1) [CORRECTED: % not 0/1]
------------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Mapping of raw stripped CSV headers to canonical snake_case column names
CANONICAL_COLUMN_MAPPING: Dict[str, str] = {
    "No of GPS Satellites Available": "gps_num_satellites",
    "Time Since Start of Day (seconds)": "timestamp_s",
    "Latitude (degrees)": "latitude_deg",
    "Longitude (degrees)": "longitude_deg",
    "Velocity (km/hr)": "gps_velocity_kmh",
    "Heading (degrees)": "gps_heading_deg",
    "Height (km)": "height_m",
    "Vertical velocity (km/hr)": "gps_vertical_velocity_kmh",
    "Sample period (seconds)": "sample_period_s",
    "Steering Angle (degrees)": "steering_angle_deg",
    "Wheel Speed Front Left (rad/sec)": "wheel_speed_fl_rad_s",
    "Wheel Speed Front Right (rad/sec)": "wheel_speed_fr_rad_s",
    "Wheel Speed Rear Left (rad/sec)": "wheel_speed_rl_rad_s",
    "Wheel Speed Rear Right (rad/sec)": "wheel_speed_rr_rad_s",
    "Yaw Rate (deg/sec)": "yaw_rate_deg_s",
    "Indicated Vehicle Speed (km/hr)": "indicated_vehicle_speed_kmh",
    "Indicated Longitudinal Acceleration (g)": "indicated_longitudinal_accel_g",
    "Indicated Lateral Acceleration (g)": "indicated_lateral_accel_g",
    "Handbrake (0 or 1)": "handbrake",
    "Gear Requested (Number fof gear employed 1-5)": "gear_requested",
    "Gear (Number fof gear employed 1-5)": "gear_actual",
    "Engine Speed (rev/min)": "engine_speed_rpm",
    "Coolant Temperature (degrees)": "coolant_temp_c",
    "Clutch Position (0 or 1)": "clutch_position",
    "Brake Pressure (psi)": "brake_pressure_psi",
    "Brake Position (0 or 1)": "brake_position",
    "Battery Voltage (volts)": "battery_voltage_v",
    "Air Temperature (degrees)": "air_temp_c",
    "Accelerator Pedal Position (0 or 1)": "accelerator_pedal_pct",
}

# Canonical dtypes mapping
CANONICAL_DTYPES: Dict[str, str] = {
    "gps_num_satellites": "int32",
    "timestamp_s": "float64",
    "latitude_deg": "float64",
    "longitude_deg": "float64",
    "gps_velocity_kmh": "float32",
    "gps_heading_deg": "float32",
    "height_m": "float32",
    "gps_vertical_velocity_kmh": "float32",
    "sample_period_s": "float32",
    "steering_angle_deg": "float32",
    "wheel_speed_fl_rad_s": "float32",
    "wheel_speed_fr_rad_s": "float32",
    "wheel_speed_rl_rad_s": "float32",
    "wheel_speed_rr_rad_s": "float32",
    "yaw_rate_deg_s": "float32",
    "indicated_vehicle_speed_kmh": "float32",
    "indicated_longitudinal_accel_g": "float32",
    "indicated_lateral_accel_g": "float32",
    "handbrake": "int8",
    "gear_requested": "int8",
    "gear_actual": "int8",
    "engine_speed_rpm": "float32",
    "coolant_temp_c": "float32",
    "clutch_position": "int8",
    "brake_pressure_psi": "float32",
    "brake_position": "int8",
    "battery_voltage_v": "float32",
    "air_temp_c": "float32",
    "accelerator_pedal_pct": "float32",
}

CANONICAL_COLUMN_ORDER: List[str] = list(CANONICAL_COLUMN_MAPPING.values())


class TimestampViolationError(ValueError):
    """Raised when timestamp sequence violates monotonicity or sample-rate assertions."""

    def __init__(self, message: str, file_path: Optional[Path] = None, row_index: Optional[int] = None):
        super().__init__(message)
        self.file_path = file_path
        self.row_index = row_index


@dataclass
class CanRunManifestEntry:
    """Metadata and audit status entry for a single unique vehicle/CAN run."""

    run_id: str
    canonical_source_path: str
    duplicate_source_paths: str = ""
    n_rows: int = 0
    duration_s: float = 0.0
    distance_km: float = 0.0
    quality_flags: str = "none"
    parse_status: str = "pending"
    parse_error: str = ""


def normalize_can_run_id(filename: str) -> str:
    """Normalize a raw CAN CSV filename or stem into a canonical run identifier.

    Examples:
        'V-M.csv' -> 'V-M'
        'V-vta10.csv' -> 'V-Vta10'
        'V-Vtb1.csv' -> 'V-Vtb1'
    """
    stem = Path(filename).stem
    if stem.lower().startswith("v-"):
        rest = stem[2:]
        if rest.lower().startswith("v"):
            rest = "V" + rest[1:]
        elif rest.lower().startswith("st"):
            rest = "St" + rest[2:]
        elif rest.lower().startswith("s"):
            rest = "S" + rest[1:]
        elif rest.lower().startswith("m"):
            rest = "M" + rest[1:]
        elif rest.lower().startswith("y"):
            rest = "Y" + rest[1:]
        return f"V-{rest}"
    return stem


def _source_priority_score(path: Path) -> int:
    """Determine the canonical priority score for a raw source path.

    Lower score = higher priority.
    Rule: Prefer Categorised copies when both exist (carries driver/scenario metadata).
    Order:
      1. Synchronised / Categorised
      2. Unsynchronised / Categorised
      3. Synchronised / Uncategorised
      4. Unsynchronised / Uncategorised
    """
    p_str = str(path).replace("\\", "/")
    if "Synchronised" in p_str and "Categorised" in p_str:
        return 1
    if "Unsynchronised" in p_str and "Categorised" in p_str:
        return 2
    if "Synchronised" in p_str and "Uncategorised" in p_str:
        return 3
    return 4


def discover_can_runs(raw_root: Path) -> List[CanRunManifestEntry]:
    """Walk raw_root, identify unique CAN runs, and select canonical paths by priority.

    Args:
        raw_root: Root path to raw dataset (e.g. data/raw/io-vnbd).

    Returns:
        Sorted list of CanRunManifestEntry instances (one per unique CAN run).
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root directory not found: {raw_root}")

    # Discover all candidate vehicle/CAN CSV files
    can_csv_candidates: List[Path] = []
    for path in raw_root.rglob("*.csv"):
        parts_lower = [p.lower() for p in path.parts]
        if ".git" in parts_lower or "_audit_scratch" in parts_lower:
            continue
        fname = path.name
        is_can = (
            fname.startswith("V-")
            or "v-dataset" in parts_lower
            or "v dataset" in parts_lower
        )
        if is_can:
            can_csv_candidates.append(path)

    # Group candidate paths by normalized run_id
    grouped_runs: Dict[str, List[Path]] = {}
    for path in can_csv_candidates:
        run_id = normalize_can_run_id(path.name)
        grouped_runs.setdefault(run_id, []).append(path)

    entries: List[CanRunManifestEntry] = []
    for run_id in sorted(grouped_runs.keys()):
        paths = grouped_runs[run_id]
        # Sort paths by priority score, breaking ties by shorter path string
        paths_sorted = sorted(paths, key=lambda p: (_source_priority_score(p), len(str(p)), str(p)))
        canonical_path = paths_sorted[0]
        duplicates = [str(p) for p in paths_sorted[1:]]

        entries.append(
            CanRunManifestEntry(
                run_id=run_id,
                canonical_source_path=str(canonical_path),
                duplicate_source_paths="|".join(duplicates),
                parse_status="pending",
            )
        )

    return entries


def compute_haversine_distance_km(lats: np.ndarray, lons: np.ndarray) -> float:
    """Compute cumulative path distance in km over valid WGS84 GPS coordinates."""
    valid_mask = (~np.isnan(lats)) & (~np.isnan(lons)) & (lats != 0.0) & (lons != 0.0)
    vlats = lats[valid_mask]
    vlons = lons[valid_mask]
    if len(vlats) < 2:
        return 0.0

    phi1, phi2 = np.radians(vlats[:-1]), np.radians(vlats[1:])
    dphi = np.radians(vlats[1:] - vlats[:-1])
    dlambda = np.radians(vlons[1:] - vlons[:-1])

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    step_km = 6371.0 * c

    # Filter out anomalous GPS teleport spikes (> 200 km/h in 0.1s is > 0.2 km per step)
    filtered_steps = step_km[step_km < 0.2]
    return float(np.sum(filtered_steps))


def parse_can_csv(
    path: Path,
    *,
    assert_monotonic: bool = True,
    assert_sample_rate: bool = True,
    max_allowed_gap_s: Optional[float] = 600.0,
) -> pd.DataFrame:
    """Parse a single raw 29-column vehicle/CAN CSV file into a canonical typed DataFrame.

    Args:
        path: Path to raw CAN CSV file.
        assert_monotonic: If True, asserts non-decreasing timestamps (dt >= 0).
        assert_sample_rate: If True, asserts median dt is ~0.1s (nominal 10 Hz).
        max_allowed_gap_s: If provided, asserts no inter-sample gap exceeds this threshold.

    Returns:
        pandas.DataFrame with canonical column names, exact column ordering, and correct types.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or missing expected headers.
        TimestampViolationError: If timestamps violate monotonicity, sample rate, or gap limits.
    """
    if not path.exists():
        raise FileNotFoundError(f"CAN CSV file not found: {path}")

    # Read CSV using latin-1 encoding to prevent character decode errors
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    if len(df) == 0:
        raise ValueError(f"CAN CSV file is empty: {path}")

    # Strip whitespace from headers
    cleaned_headers = [c.strip() for c in df.columns]
    df.columns = cleaned_headers

    # Validate header presence against expected 29 raw columns
    missing_cols = set(CANONICAL_COLUMN_MAPPING.keys()) - set(cleaned_headers)
    if missing_cols:
        raise ValueError(f"CAN CSV missing {len(missing_cols)} expected columns in {path}: {sorted(missing_cols)}")

    # Extract and validate timestamps
    time_raw = pd.to_numeric(df["Time Since Start of Day (seconds)"], errors="coerce")
    if time_raw.isna().any():
        bad_idx = int(time_raw.isna().idxmax())
        raise TimestampViolationError(
            f"Invalid/NaN timestamp in {path} at row index {bad_idx}",
            file_path=path,
            row_index=bad_idx,
        )

    t = time_raw.values
    if len(t) > 1:
        dt = np.diff(t)

        if assert_monotonic:
            neg_indices = np.where(dt < 0.0)[0]
            if len(neg_indices) > 0:
                first_bad = int(neg_indices[0])
                raise TimestampViolationError(
                    f"Timestamp decreased at row {first_bad + 1} in {path}: "
                    f"t[{first_bad + 1}]={t[first_bad + 1]} < t[{first_bad}]={t[first_bad]} (dt={dt[first_bad]:.4f}s)",
                    file_path=path,
                    row_index=first_bad + 1,
                )

        if assert_sample_rate:
            median_dt = float(np.median(dt))
            if not (0.05 <= median_dt <= 0.15):
                raise TimestampViolationError(
                    f"Sample rate deviates from 10 Hz in {path}: median dt = {median_dt:.4f}s (expected ~0.10s)",
                    file_path=path,
                    row_index=0,
                )

        if max_allowed_gap_s is not None:
            gap_indices = np.where(dt > max_allowed_gap_s)[0]
            if len(gap_indices) > 0:
                first_gap = int(gap_indices[0])
                raise TimestampViolationError(
                    f"Timestamp gap of {dt[first_gap]:.2f}s at row {first_gap + 1} exceeds max allowed {max_allowed_gap_s}s in {path}",
                    file_path=path,
                    row_index=first_gap + 1,
                )

    # Rename columns to canonical snake_case
    df_canonical = df.rename(columns=CANONICAL_COLUMN_MAPPING)

    # Reorder columns to fixed canonical order
    df_canonical = df_canonical[CANONICAL_COLUMN_ORDER]

    # Cast dtypes strictly
    for col, dtype_str in CANONICAL_DTYPES.items():
        if dtype_str.startswith("int"):
            df_canonical[col] = pd.to_numeric(df_canonical[col], errors="raise").round().astype(dtype_str)
        elif dtype_str.startswith("float"):
            df_canonical[col] = pd.to_numeric(df_canonical[col], errors="raise").astype(dtype_str)

    return df_canonical


def detect_quality_flags(df: pd.DataFrame) -> List[str]:
    """Inspect parsed CAN dataframe for known sensor anomalies and degraded channels."""
    flags: List[str] = []

    # Check for dead wheel speed / stationary run
    ws_cols = [
        "wheel_speed_fl_rad_s",
        "wheel_speed_fr_rad_s",
        "wheel_speed_rl_rad_s",
        "wheel_speed_rr_rad_s",
    ]
    all_zero_ws = all((df[c] == 0.0).all() for c in ws_cols)
    if all_zero_ws:
        flags.append("zero_wheel_speed")

    # Check for zero steering angle
    if (df["steering_angle_deg"] == 0.0).all():
        flags.append("zero_steering")

    # Check for large timestamp gaps
    t = df["timestamp_s"].values
    if len(t) > 1:
        dt = np.diff(t)
        if (dt > 1.0).any():
            flags.append("timestamp_gap")

    return flags


def write_parquet(df: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    """Write DataFrame to deterministic snappy-compressed Parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.parquet"

    # Use pyarrow with deterministic settings (no index column, consistent schema)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    return out_path


def ingest_all_can_runs(raw_root: Path, out_dir: Path) -> List[CanRunManifestEntry]:
    """Execute complete ingestion pipeline for all discovered CAN runs.

    Args:
        raw_root: Directory containing raw dataset.
        out_dir: Destination directory for processed Parquet files and manifest.csv.

    Returns:
        List of completed CanRunManifestEntry instances.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_can_runs(raw_root)

    manifest_records = []
    for entry in entries:
        source_path = Path(entry.canonical_source_path)
        try:
            df = parse_can_csv(source_path)
            entry.n_rows = len(df)
            t = df["timestamp_s"].values
            entry.duration_s = float(t[-1] - t[0]) if len(t) > 1 else 0.0

            lats = df["latitude_deg"].values
            lons = df["longitude_deg"].values
            entry.distance_km = compute_haversine_distance_km(lats, lons)

            flags = detect_quality_flags(df)
            entry.quality_flags = "|".join(flags) if flags else "none"

            # Write Parquet
            write_parquet(df, entry.run_id, out_dir)
            entry.parse_status = "ok"
            entry.parse_error = ""

        except Exception as exc:
            entry.parse_status = "failed"
            entry.parse_error = str(exc)

        manifest_records.append(entry)

    # Write manifest.csv
    manifest_path = out_dir / "manifest.csv"
    fieldnames = [
        "run_id",
        "canonical_source_path",
        "duplicate_source_paths",
        "n_rows",
        "duration_s",
        "distance_km",
        "quality_flags",
        "parse_status",
        "parse_error",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in manifest_records:
            writer.writerow({
                "run_id": item.run_id,
                "canonical_source_path": item.canonical_source_path,
                "duplicate_source_paths": item.duplicate_source_paths,
                "n_rows": item.n_rows,
                "duration_s": f"{item.duration_s:.3f}",
                "distance_km": f"{item.distance_km:.3f}",
                "quality_flags": item.quality_flags,
                "parse_status": item.parse_status,
                "parse_error": item.parse_error,
            })

    return manifest_records
