"""Smartphone dataset ingestion module for IO-VNBD.

This module provides discovery, deduplication, schema-variant reconciliation,
typed casting, unit correction, validation, and Parquet serialization for the
five smartphone CSV schema variants.

Canonical Column Schema Definition:
---------------------------------------------------------------------------------------------------------
Canonical Column Name        Dtype      Unit       Source Raw Header(s) / Notes
---------------------------------------------------------------------------------------------------------
timestamp_ms                 int64      ms         TIME SINCE START (ms)
timestamp_s                  float64    s          Derived: timestamp_ms / 1000.0
date_utc                     string     ISO text   DATE (YYYY-MO-DD HH-MI-SS_SSS...)
latitude_deg                 float64    deg        GPS LATITUDE (degrees)
longitude_deg                float64    deg        GPS LONGITUDE (degrees)
altitude_m                   float32    m          GPS ALTITUDE (m)
gps_speed_raw                float32    raw        GPS SPEED (Kmh) raw log value
gps_speed_ms                 float32    m/s        True speed in m/s (raw / 3.6 for km/h runs, raw for m/s)
gps_speed_kmh                float32    km/h       True speed in km/h (raw for km/h runs, raw * 3.6 for m/s)
gps_accuracy_m               float32    m          GPS ACCURACY (m)
gps_bearing_deg              float32    deg        GPS ORIENTATION (°)
gps_satellites_in_range      Int16      count      GPS SATELLITES IN RANGE [nullable int; handles Excel dates]
accel_x_m_s2                 float32    m/s²       ACCELEROMETER X (m/s²)
accel_y_m_s2                 float32    m/s²       ACCELEROMETER Y (m/s²)
accel_z_m_s2                 float32    m/s²       ACCELEROMETER Z (m/s²)
gravity_x_m_s2               float32    m/s²       GRAVITY X (m/s²)
gravity_y_m_s2               float32    m/s²       GRAVITY Y (m/s²)
gravity_z_m_s2               float32    m/s²       GRAVITY Z (m/s²)
gyro_x_rad_s                 float32    rad/s      GYROSCOPE X (rad/s) or GYROSCOPE Yaw (rad/s)
gyro_y_rad_s                 float32    rad/s      GYROSCOPE Y (rad/s) or GYROSCOPE Pitch (rad/s)
gyro_z_rad_s                 float32    rad/s      GYROSCOPE Z (rad/s) or GYROSCOPE Roll (rad/s)
mag_x_uT                     float32    µT         MAGNETIC FIELD X (μT) [null in French 18-col runs]
mag_y_uT                     float32    µT         MAGNETIC FIELD Y (μT) [null in French 18-col runs]
mag_z_uT                     float32    µT         MAGNETIC FIELD Z (μT) [null in French 18-col runs]
orientation_azimuth_deg      float32    deg        ORIENTATION (Azimuth/Yaw) (°) [null in French 18-col runs]
orientation_pitch_deg        float32    deg        ORIENTATION (Pitch) (°) [null in French 18-col runs]
orientation_roll_deg         float32    deg        ORIENTATION (Roll ) (°) [null in French 18-col runs]
---------------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Standard 27 canonical column ordering
CANONICAL_COLUMN_ORDER: List[str] = [
    "timestamp_ms",
    "timestamp_s",
    "date_utc",
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
    "gps_speed_raw",
    "gps_speed_ms",
    "gps_speed_kmh",
    "gps_accuracy_m",
    "gps_bearing_deg",
    "gps_satellites_in_range",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gravity_x_m_s2",
    "gravity_y_m_s2",
    "gravity_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "mag_x_uT",
    "mag_y_uT",
    "mag_z_uT",
    "orientation_azimuth_deg",
    "orientation_pitch_deg",
    "orientation_roll_deg",
]

# Runs where raw speed was recorded in km/h rather than m/s
KNOWN_RAW_KMH_RUNS: Set[str] = {"s-a2", "s-a9", "s-a10"}


@dataclass
class PhoneRunManifestEntry:
    """Metadata and audit status entry for a single unique smartphone run."""

    run_id: str
    canonical_source_path: str
    duplicate_source_paths: str = ""
    schema_variant: str = "unknown"
    n_rows: int = 0
    duration_s: float = 0.0
    distance_km: float = 0.0
    median_dt_s: float = 0.0
    sample_rate_class: str = "10hz"
    quality_flags: str = "none"
    parse_status: str = "pending"
    parse_error: str = ""


def normalize_phone_run_id(filename: str) -> str:
    """Normalize a raw smartphone CSV filename or stem into a canonical run identifier.

    Examples:
        'S-M.csv' -> 'S-M'
        'S-s1.csv' -> 'S-S1'
        'S-vta10.csv' -> 'S-Vta10'
    """
    stem = Path(filename).stem
    if stem.lower().startswith("s-"):
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
        elif rest.lower().startswith("a"):
            rest = "A" + rest[1:]
        elif rest.lower().startswith("t"):
            rest = "T" + rest[1:]
        elif rest.lower().startswith("i"):
            rest = "I" + rest[1:]
        return f"S-{rest}"
    return stem


def _source_priority_score(path: Path) -> int:
    """Determine the canonical priority score for a raw source path.

    Lower score = higher priority. Prefer Categorised copies.
    """
    p_str = str(path).replace("\\", "/")
    if "Synchronised" in p_str and "Categorised" in p_str:
        return 1
    if "Unsynchronised" in p_str and "Categorised" in p_str:
        return 2
    if "Synchronised" in p_str and "Uncategorised" in p_str:
        return 3
    return 4


def detect_schema_variant(path: Path) -> str:
    """Inspect raw CSV header and classify into one of the 5 schema variants.

    Returns:
        One of: 'standard_24', 'typo_header_24', 'truncated_18', 'corrupted_delimiter_25'.
    """
    with open(path, "r", encoding="latin-1") as f:
        header_line = f.readline()

    raw_tokens = [t.strip() for t in header_line.split(",")]

    # Check for S-A4 corrupted delimiter: trailing comma produces 25 tokens with empty end
    if len(raw_tokens) >= 25 and (raw_tokens[-1] == "" or "Unnamed" in raw_tokens[-1]):
        return "corrupted_delimiter_25"

    # Check for French 18-column truncated schema
    if len(raw_tokens) == 18:
        return "truncated_18"

    # Check for missing closing parenthesis typo in DATE header
    if any(t.startswith("DATE (YYYY-MO-DD") and not t.endswith(")") for t in raw_tokens):
        return "typo_header_24"

    return "standard_24"


def discover_phone_runs(raw_root: Path) -> List[PhoneRunManifestEntry]:
    """Walk raw_root, identify unique smartphone runs, and select canonical paths by priority.

    Args:
        raw_root: Root path to raw dataset.

    Returns:
        Sorted list of PhoneRunManifestEntry instances (one per unique smartphone run).
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root directory not found: {raw_root}")

    phone_csv_candidates: List[Path] = []
    for path in raw_root.rglob("*.csv"):
        parts_lower = [p.lower() for p in path.parts]
        if ".git" in parts_lower or "_audit_scratch" in parts_lower:
            continue
        fname = path.name
        is_phone = (
            fname.startswith("S-")
            or "s-dataset" in parts_lower
            or "s dataset" in parts_lower
        )
        if is_phone:
            phone_csv_candidates.append(path)

    grouped_runs: Dict[str, List[Path]] = {}
    for path in phone_csv_candidates:
        run_id = normalize_phone_run_id(path.name)
        grouped_runs.setdefault(run_id, []).append(path)

    entries: List[PhoneRunManifestEntry] = []
    for run_id in sorted(grouped_runs.keys()):
        paths = grouped_runs[run_id]
        paths_sorted = sorted(paths, key=lambda p: (_source_priority_score(p), len(str(p)), str(p)))
        canonical_path = paths_sorted[0]
        duplicates = [str(p) for p in paths_sorted[1:]]

        variant = detect_schema_variant(canonical_path)

        entries.append(
            PhoneRunManifestEntry(
                run_id=run_id,
                canonical_source_path=str(canonical_path),
                duplicate_source_paths="|".join(duplicates),
                schema_variant=variant,
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

    # Filter out anomalous GPS teleport spikes (> 0.5 km per sample)
    filtered_steps = step_km[step_km < 0.5]
    return float(np.sum(filtered_steps))


def _parse_satellite_count(val: object) -> Optional[int]:
    """Extract integer count of satellites in range from strings like '18 / 19', handling Excel date mangling."""
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if not val_str:
        return None
    # Match integer prefix before slash: e.g. '18 / 19' -> 18
    m = re.match(r"^(\d+)", val_str)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    # Excel date strings like 'Aug-20' return None
    return None


def parse_phone_csv(
    path: Path,
    variant: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """Parse raw smartphone CSV of any variant into a unified 27-column typed DataFrame.

    Args:
        path: Path to raw smartphone CSV file.
        variant: Optional pre-detected schema variant. Detected automatically if None.

    Returns:
        Tuple of (canonical_dataframe, median_dt_seconds).
    """
    if not path.exists():
        raise FileNotFoundError(f"Smartphone CSV file not found: {path}")

    if variant is None:
        variant = detect_schema_variant(path)

    # Read and normalize rows based on variant
    if variant == "corrupted_delimiter_25":
        # Repair S-A4.csv: strip empty token at index 6 on each data row
        cleaned_rows = []
        with open(path, "r", encoding="latin-1") as f:
            header_raw = f.readline().strip().rstrip(",")
            header_tokens = [t.strip() for t in header_raw.split(",")]
            # Standardize header length
            cleaned_rows.append(header_tokens[:24])
            for line in f:
                line_str = line.strip().rstrip(",")
                if not line_str:
                    continue
                toks = [t.strip() for t in line_str.split(",")]
                if len(toks) >= 25 and toks[6] == "":
                    toks = toks[:6] + toks[7:]
                cleaned_rows.append(toks[:24])

        output_buffer = io.StringIO()
        csv_writer = csv.writer(output_buffer)
        csv_writer.writerows(cleaned_rows)
        output_buffer.seek(0)
        df = pd.read_csv(output_buffer, low_memory=False)

    else:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    if len(df) == 0:
        raise ValueError(f"Smartphone CSV file is empty: {path}")

    # Strip whitespace from headers
    df.columns = [c.strip() for c in df.columns]

    n_rows = len(df)
    out_dict = {}

    # Extract timestamps
    time_col_idx = 7
    t_ms_raw = pd.to_numeric(df.iloc[:, time_col_idx], errors="coerce").fillna(0).astype(np.int64)
    out_dict["timestamp_ms"] = t_ms_raw
    out_dict["timestamp_s"] = (t_ms_raw / 1000.0).astype(np.float64)

    # Compute median dt in seconds
    t_vals = t_ms_raw.values
    if len(t_vals) > 1:
        dt_s = np.diff(t_vals) / 1000.0
        # Ignore zero dt or extreme pauses when estimating nominal median
        valid_dt = dt_s[dt_s > 0]
        median_dt = float(np.median(valid_dt)) if len(valid_dt) > 0 else 0.1000
    else:
        median_dt = 0.1000

    # Date column (col index 8)
    date_col_idx = 8
    out_dict["date_utc"] = df.iloc[:, date_col_idx].astype(str)

    # GPS coordinates
    out_dict["latitude_deg"] = pd.to_numeric(df.iloc[:, 0], errors="coerce").astype(np.float64)
    out_dict["longitude_deg"] = pd.to_numeric(df.iloc[:, 1], errors="coerce").astype(np.float64)
    out_dict["altitude_m"] = pd.to_numeric(df.iloc[:, 2], errors="coerce").astype(np.float32)

    # Speed handling: check if raw speed is km/h or m/s
    stem_lower = path.stem.lower()
    raw_speed = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0.0).astype(np.float32)
    out_dict["gps_speed_raw"] = raw_speed

    is_raw_kmh = (stem_lower in KNOWN_RAW_KMH_RUNS) or (raw_speed.max() > 60.0)
    if is_raw_kmh:
        # Raw value was recorded in km/h
        out_dict["gps_speed_kmh"] = raw_speed
        out_dict["gps_speed_ms"] = (raw_speed / 3.6).astype(np.float32)
    else:
        # Raw value was recorded in m/s (standard Android Location.getSpeed())
        out_dict["gps_speed_ms"] = raw_speed
        out_dict["gps_speed_kmh"] = (raw_speed * 3.6).astype(np.float32)

    out_dict["gps_accuracy_m"] = pd.to_numeric(df.iloc[:, 4], errors="coerce").astype(np.float32)
    out_dict["gps_bearing_deg"] = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(np.float32)

    # Satellite count (col index 6)
    sat_raw = df.iloc[:, 6]
    sat_clean = [_parse_satellite_count(v) for v in sat_raw]
    out_dict["gps_satellites_in_range"] = pd.Series(sat_clean, dtype="Int16")

    # Accelerometer (col indices 9, 10, 11)
    out_dict["accel_x_m_s2"] = pd.to_numeric(df.iloc[:, 9], errors="coerce").astype(np.float32)
    out_dict["accel_y_m_s2"] = pd.to_numeric(df.iloc[:, 10], errors="coerce").astype(np.float32)
    out_dict["accel_z_m_s2"] = pd.to_numeric(df.iloc[:, 11], errors="coerce").astype(np.float32)

    # Gravity (col indices 12, 13, 14)
    out_dict["gravity_x_m_s2"] = pd.to_numeric(df.iloc[:, 12], errors="coerce").astype(np.float32)
    out_dict["gravity_y_m_s2"] = pd.to_numeric(df.iloc[:, 13], errors="coerce").astype(np.float32)
    out_dict["gravity_z_m_s2"] = pd.to_numeric(df.iloc[:, 14], errors="coerce").astype(np.float32)

    # Gyroscope (col indices 15, 16, 17)
    out_dict["gyro_x_rad_s"] = pd.to_numeric(df.iloc[:, 15], errors="coerce").astype(np.float32)
    out_dict["gyro_y_rad_s"] = pd.to_numeric(df.iloc[:, 16], errors="coerce").astype(np.float32)
    out_dict["gyro_z_rad_s"] = pd.to_numeric(df.iloc[:, 17], errors="coerce").astype(np.float32)

    # Magnetometer and Orientation: Present in 24-col variants, absent in 18-col variant
    if variant == "truncated_18" or df.shape[1] <= 18:
        # Explicit genuine nulls (never zero-filled)
        nan_series = pd.Series([np.nan] * n_rows, dtype=np.float32)
        out_dict["mag_x_uT"] = nan_series
        out_dict["mag_y_uT"] = nan_series
        out_dict["mag_z_uT"] = nan_series
        out_dict["orientation_azimuth_deg"] = nan_series
        out_dict["orientation_pitch_deg"] = nan_series
        out_dict["orientation_roll_deg"] = nan_series
    else:
        out_dict["mag_x_uT"] = pd.to_numeric(df.iloc[:, 18], errors="coerce").astype(np.float32)
        out_dict["mag_y_uT"] = pd.to_numeric(df.iloc[:, 19], errors="coerce").astype(np.float32)
        out_dict["mag_z_uT"] = pd.to_numeric(df.iloc[:, 20], errors="coerce").astype(np.float32)
        out_dict["orientation_azimuth_deg"] = pd.to_numeric(df.iloc[:, 21], errors="coerce").astype(np.float32)
        out_dict["orientation_pitch_deg"] = pd.to_numeric(df.iloc[:, 22], errors="coerce").astype(np.float32)
        out_dict["orientation_roll_deg"] = pd.to_numeric(df.iloc[:, 23], errors="coerce").astype(np.float32)

    df_canonical = pd.DataFrame(out_dict, columns=CANONICAL_COLUMN_ORDER)
    return df_canonical, median_dt


def detect_phone_quality_flags(
    df: pd.DataFrame,
    variant: str,
    median_dt: float,
    stem: str,
) -> Tuple[str, List[str]]:
    """Determine sample rate class and quality flags for a smartphone run."""
    flags: List[str] = []

    # Classify sample rate
    if 0.40 <= median_dt <= 0.60 or stem.lower() in {f"s-a{i}" for i in [1, 2, 3, 9, 10, 11, 12, 13]}:
        rate_class = "2hz"
        flags.append("low_sample_rate")
    elif median_dt <= 0.02 or stem.lower() in {"s-t1", "s-t4", "s-t5", "s-t6"}:
        rate_class = "burst"
        flags.append("burst_sampling")
    else:
        rate_class = "10hz"

    # Missing magnetometer
    if variant == "truncated_18" or df["mag_x_uT"].isna().all():
        flags.append("no_magnetometer")

    # Delimiter repair
    if variant == "corrupted_delimiter_25":
        flags.append("delimiter_repaired")

    # Raw speed unit
    if stem.lower() in KNOWN_RAW_KMH_RUNS or df["gps_speed_raw"].max() > 60.0:
        flags.append("raw_speed_kmh")

    # Clock resets (negative dt)
    t = df["timestamp_ms"].values
    if len(t) > 1:
        dt = np.diff(t)
        if (dt < 0).any():
            flags.append("clock_reset")

        # Internal multi-minute recording gaps (> 60s)
        if (dt > 60000).any():
            flags.append("internal_gap")

    return rate_class, flags


def write_parquet(df: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    """Write DataFrame to deterministic snappy-compressed Parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.parquet"

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    return out_path


def ingest_all_phone_runs(raw_root: Path, out_dir: Path) -> List[PhoneRunManifestEntry]:
    """Execute complete ingestion pipeline for all discovered smartphone runs.

    Args:
        raw_root: Directory containing raw dataset.
        out_dir: Destination directory for processed Parquet files and manifest.csv.

    Returns:
        List of completed PhoneRunManifestEntry instances.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_phone_runs(raw_root)

    manifest_records = []
    for entry in entries:
        source_path = Path(entry.canonical_source_path)
        try:
            df, median_dt = parse_phone_csv(source_path, variant=entry.schema_variant)
            entry.n_rows = len(df)
            entry.median_dt_s = median_dt

            t_ms = df["timestamp_ms"].values
            entry.duration_s = float((t_ms[-1] - t_ms[0]) / 1000.0) if len(t_ms) > 1 else 0.0

            lats = df["latitude_deg"].values
            lons = df["longitude_deg"].values
            entry.distance_km = compute_haversine_distance_km(lats, lons)

            rate_class, flags = detect_phone_quality_flags(
                df,
                variant=entry.schema_variant,
                median_dt=median_dt,
                stem=source_path.stem,
            )
            entry.sample_rate_class = rate_class
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
        "schema_variant",
        "n_rows",
        "duration_s",
        "distance_km",
        "median_dt_s",
        "sample_rate_class",
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
                "schema_variant": item.schema_variant,
                "n_rows": item.n_rows,
                "duration_s": f"{item.duration_s:.3f}",
                "distance_km": f"{item.distance_km:.3f}",
                "median_dt_s": f"{item.median_dt_s:.4f}",
                "sample_rate_class": item.sample_rate_class,
                "quality_flags": item.quality_flags,
                "parse_status": item.parse_status,
                "parse_error": item.parse_error,
            })

    return manifest_records
