"""Loader and validator for C++ strapdown INS replay trajectory predictions.

Reads prediction CSV files from data/processed/cpp_predictions/ and validates
them against outage window specifications and ground truth formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp_s",
    "latitude_deg",
    "longitude_deg",
    "speed_ms",
    "heading_deg",
]


def load_cpp_prediction(csv_path: Union[Path, str]) -> pd.DataFrame:
    """Load a C++ strapdown INS trajectory prediction CSV file.

    Args:
        csv_path: Path to the prediction CSV file.

    Returns:
        DataFrame with columns: timestamp_s, latitude_deg, longitude_deg, speed_ms, heading_deg.

    Raises:
        FileNotFoundError: If the prediction file does not exist.
        ValueError: If the file is empty, has missing columns, or contains < 2 points.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Prediction file is empty: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Prediction file {path.name} missing required columns: {missing_cols}. Found: {df.columns.tolist()}"
        )

    if len(df) < 2:
        raise ValueError(
            f"Prediction file {path.name} contains fewer than 2 points ({len(df)} rows)."
        )

    # Cast types strictly
    df = df[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check that coordinate columns are not entirely NaN
    if df["latitude_deg"].isna().all() or df["longitude_deg"].isna().all():
        raise ValueError(f"Prediction file {path.name} contains all-NaN coordinate values.")

    return df


def validate_prediction_against_manifest(
    pred_df: pd.DataFrame,
    manifest_row: Union[pd.Series, Dict[str, Any]],
    start_tolerance_s: float = 0.5,
    duration_tolerance_s: float = 1.0,
) -> Dict[str, Any]:
    """Validate a loaded prediction DataFrame against its manifest metadata.

    Args:
        pred_df: DataFrame loaded via load_cpp_prediction.
        manifest_row: Row from manifest.csv as Series or Dict.
        start_tolerance_s: Max allowed difference between first timestamp and start_s (seconds).
        duration_tolerance_s: Max allowed difference between trajectory span and outage_s (seconds).

    Returns:
        Dictionary with keys:
            - 'valid': bool
            - 'n_rows': int
            - 't_start': float
            - 't_end': float
            - 'span_s': float
            - 'expected_outage_s': float
            - 'issues': List[str]
    """
    issues: List[str] = []
    n_rows = len(pred_df)

    if n_rows < 2:
        issues.append(f"Insufficient points: {n_rows} rows < 2 minimum.")

    t_start = float(pred_df["timestamp_s"].iloc[0])
    t_end = float(pred_df["timestamp_s"].iloc[-1])
    span_s = t_end - t_start

    expected_start = float(manifest_row["start_s"])
    expected_outage_s = float(manifest_row["outage_s"])
    expected_end = float(manifest_row["end_s"])

    # Check start timestamp proximity
    if abs(t_start - expected_start) > start_tolerance_s:
        issues.append(
            f"Start timestamp mismatch: trajectory starts at {t_start:.3f}s, expected ~{expected_start:.3f}s "
            f"(diff {abs(t_start - expected_start):.3f}s > {start_tolerance_s:.3f}s tolerance)."
        )

    # Check span vs expected outage duration
    if abs(span_s - expected_outage_s) > duration_tolerance_s:
        issues.append(
            f"Duration mismatch: trajectory spans {span_s:.3f}s, expected ~{expected_outage_s:.1f}s "
            f"(diff {abs(span_s - expected_outage_s):.3f}s > {duration_tolerance_s:.3f}s tolerance)."
        )

    # Check for excessive NaNs
    lat_nans = int(pred_df["latitude_deg"].isna().sum())
    lon_nans = int(pred_df["longitude_deg"].isna().sum())
    if lat_nans > 0 or lon_nans > 0:
        issues.append(f"Contains null coordinates: {lat_nans} lat NaNs, {lon_nans} lon NaNs.")

    return {
        "valid": len(issues) == 0,
        "n_rows": n_rows,
        "t_start": t_start,
        "t_end": t_end,
        "span_s": span_s,
        "expected_outage_s": expected_outage_s,
        "issues": issues,
    }


def spot_check_predictions(
    manifest_df: pd.DataFrame,
    pred_dir: Union[Path, str],
    sample_per_duration: int = 2,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Spot-check a stratified sample of prediction files against manifest metadata.

    Args:
        manifest_df: Full or filtered manifest DataFrame.
        pred_dir: Directory containing prediction CSV files.
        sample_per_duration: Number of instances to check per duration (10, 30, 60, 120, 180).
        random_seed: Random state for deterministic sampling.

    Returns:
        List of spot-check result dictionaries.
    """
    pred_path = Path(pred_dir)
    ok_manifest = manifest_df[manifest_df["status"] == "ok"].copy()
    durations = [10, 30, 60, 120, 180]
    results: List[Dict[str, Any]] = []

    for d in durations:
        subset = ok_manifest[ok_manifest["outage_s"] == d]
        if subset.empty:
            continue
        sample = subset.sample(min(len(subset), sample_per_duration), random_state=random_seed)
        for _, row in sample.iterrows():
            outage_id = str(row["outage_id"]).strip()
            csv_file = pred_path / f"{outage_id}.csv"

            if not csv_file.exists():
                results.append({
                    "outage_id": outage_id,
                    "outage_s": d,
                    "split": row["split"],
                    "valid": False,
                    "issues": [f"File missing: {csv_file.name}"],
                })
                continue

            try:
                df = load_cpp_prediction(csv_file)
                val = validate_prediction_against_manifest(df, row)
                results.append({
                    "outage_id": outage_id,
                    "outage_s": d,
                    "split": row["split"],
                    "valid": val["valid"],
                    "n_rows": val["n_rows"],
                    "span_s": val["span_s"],
                    "issues": val["issues"],
                })
            except Exception as e:
                results.append({
                    "outage_id": outage_id,
                    "outage_s": d,
                    "split": row["split"],
                    "valid": False,
                    "issues": [f"Load error: {e}"],
                })

    return results
