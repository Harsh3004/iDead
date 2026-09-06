"""Unit tests for vehicle/CAN dataset ingestion and parsing."""

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from dataeval.ingest.vehicle_can import (
    CANONICAL_COLUMN_MAPPING,
    CANONICAL_COLUMN_ORDER,
    CANONICAL_DTYPES,
    TimestampViolationError,
    detect_quality_flags,
    discover_can_runs,
    normalize_can_run_id,
    parse_can_csv,
    write_parquet,
)


def _make_synthetic_can_csv(path: Path, n_rows: int = 20, **kwargs) -> Path:
    """Helper to generate a valid synthetic 29-column CAN CSV with configurable column values."""
    times = kwargs.get("times", [29600.0 + i * 0.1 for i in range(n_rows)])
    lats = kwargs.get("lats", [52.4 + i * 0.0001 for i in range(n_rows)])
    lons = kwargs.get("lons", [-1.5 + i * 0.0001 for i in range(n_rows)])
    heights = kwargs.get("heights", [120.5 for _ in range(n_rows)])
    pedals = kwargs.get("pedals", [25.0 for _ in range(n_rows)])
    ws_fl = kwargs.get("ws_fl", [15.0 for _ in range(n_rows)])
    steer = kwargs.get("steer", [10.0 for _ in range(n_rows)])

    data = {
        "No of GPS Satellites Available": [136.0] * n_rows,
        "Time Since Start of Day (seconds)": times,
        "Latitude (degrees)": lats,
        "Longitude (degrees)": lons,
        "Velocity (km/hr)": [45.0] * n_rows,
        "Heading (degrees)": [90.0] * n_rows,
        "Height (km)": heights,
        "Vertical velocity (km/hr)": [0.0] * n_rows,
        "Sample period (seconds)": [0.1] * n_rows,
        "Steering Angle (degrees)": steer,
        "Wheel Speed Front Left (rad/sec)": ws_fl,
        "Wheel Speed Front Right (rad/sec)": ws_fl,
        "Wheel Speed Rear Left (rad/sec)": ws_fl,
        "Wheel Speed Rear Right (rad/sec)": ws_fl,
        "Yaw Rate (deg/sec)": [0.0] * n_rows,
        "Indicated Vehicle Speed (km/hr)": [45.0] * n_rows,
        "Indicated Longitudinal Acceleration (g)": [0.05] * n_rows,
        "Indicated Lateral Acceleration (g)": [0.01] * n_rows,
        "Handbrake (0 or 1)": [0.0] * n_rows,
        "Gear Requested (Number fof gear employed 1-5)": [3.0] * n_rows,
        "Gear (Number fof gear employed 1-5)": [3.0] * n_rows,
        "Engine Speed (rev/min)": [2100.0] * n_rows,
        "Coolant Temperature (degrees)": [85.0] * n_rows,
        "Clutch Position (0 or 1)": [0.0] * n_rows,
        "Brake Pressure (psi)": [0.0] * n_rows,
        "Brake Position (0 or 1)": [0.0] * n_rows,
        "Battery Voltage (volts)": [14.2] * n_rows,
        "Air Temperature (degrees)": [18.5] * n_rows,
        "Accelerator Pedal Position (0 or 1)": pedals,
    }
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


class TestVehicleCanIngest(unittest.TestCase):
    """Test suite for vehicle/CAN CSV parsing, schema normalization, and Parquet writing."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_column_renaming_and_types(self):
        """Verify column names are normalized to canonical snake_case and dtypes are correctly cast."""
        csv_file = self.temp_dir / "test_run.csv"
        _make_synthetic_can_csv(csv_file)

        df_parsed = parse_can_csv(csv_file)

        # All 29 canonical columns present in exact order
        self.assertEqual(list(df_parsed.columns), CANONICAL_COLUMN_ORDER)
        self.assertEqual(len(df_parsed.columns), 29)

        # Typo normalization check
        self.assertIn("gear_requested", df_parsed.columns)
        self.assertIn("gear_actual", df_parsed.columns)
        self.assertNotIn("Gear Requested (Number fof gear employed 1-5)", df_parsed.columns)
        self.assertNotIn("Gear (Number fof gear employed 1-5)", df_parsed.columns)

        # Check dtypes
        for col, expected_dtype in CANONICAL_DTYPES.items():
            actual_dtype = str(df_parsed[col].dtype)
            self.assertEqual(
                actual_dtype,
                expected_dtype,
                f"Dtype mismatch on {col}: expected {expected_dtype}, got {actual_dtype}",
            )

    def test_units_height_and_accelerator(self):
        """Verify height_m and accelerator_pedal_pct are parsed as plausible floats, not booleans or km."""
        csv_file = self.temp_dir / "test_units.csv"
        _make_synthetic_can_csv(csv_file, heights=[142.75] * 10, pedals=[78.5] * 10, n_rows=10)

        df = parse_can_csv(csv_file)

        # Height in meters
        self.assertEqual(df["height_m"].dtype, np.float32)
        self.assertTrue(np.isclose(df["height_m"].iloc[0], 142.75))
        self.assertTrue(20.0 < df["height_m"].iloc[0] < 600.0)

        # Accelerator pedal percentage
        self.assertEqual(df["accelerator_pedal_pct"].dtype, np.float32)
        self.assertTrue(np.isclose(df["accelerator_pedal_pct"].iloc[0], 78.5))
        self.assertTrue(df["accelerator_pedal_pct"].iloc[0] > 1.0)

    def test_timestamp_violation_negative_dt(self):
        """Verify negative dt (non-monotonic) raises TimestampViolationError with row index."""
        csv_file = self.temp_dir / "test_bad_monotonic.csv"
        # Row index 3 decreases timestamp: 29600.0 (row 0), 29600.1 (row 1), 29600.2 (row 2), 29600.05 (row 3)
        times = [29600.0, 29600.1, 29600.2, 29600.05, 29600.4]
        _make_synthetic_can_csv(csv_file, n_rows=len(times), times=times)

        with self.assertRaises(TimestampViolationError) as ctx:
            parse_can_csv(csv_file)

        self.assertIn("decreased at row 3", str(ctx.exception))
        self.assertEqual(ctx.exception.row_index, 3)

    def test_timestamp_violation_excessive_gap(self):
        """Verify inter-sample gap exceeding threshold raises TimestampViolationError."""
        csv_file = self.temp_dir / "test_bad_gap.csv"
        # Gap of 15.0s between row index 1 and row index 2
        times = [29600.0, 29600.1, 29615.1, 29615.2]
        _make_synthetic_can_csv(csv_file, n_rows=len(times), times=times)

        with self.assertRaises(TimestampViolationError) as ctx:
            parse_can_csv(csv_file, max_allowed_gap_s=10.0)

        self.assertIn("exceeds max allowed 10.0s", str(ctx.exception))
        self.assertEqual(ctx.exception.row_index, 2)

    def test_dedup_prefers_categorised(self):
        """Verify discover_can_runs prioritizes Categorised copies over Uncategorised copies."""
        raw_root = self.temp_dir / "raw_dataset"

        cat_path = (
            raw_root
            / "Synchronised V abd S datasets"
            / "Categorised IOVNB Dataset"
            / "S (Driver A)"
            / "S1"
            / "V-S1.csv"
        )
        uncat_path = (
            raw_root
            / "Synchronised V abd S datasets"
            / "Uncategorised IOVNB Dataset"
            / "V-Dataset"
            / "V-S1.csv"
        )

        _make_synthetic_can_csv(cat_path)
        _make_synthetic_can_csv(uncat_path)

        entries = discover_can_runs(raw_root)
        self.assertEqual(len(entries), 1)

        entry = entries[0]
        self.assertEqual(entry.run_id, "V-S1")
        self.assertIn("Categorised", entry.canonical_source_path)
        self.assertIn("Uncategorised", entry.duplicate_source_paths)

    def test_quality_flags_detection(self):
        """Verify zero_wheel_speed and zero_steering flags are detected."""
        csv_zero_ws = self.temp_dir / "zero_ws.csv"
        _make_synthetic_can_csv(csv_zero_ws, ws_fl=[0.0] * 10, steer=[5.0] * 10, n_rows=10)
        df_ws = parse_can_csv(csv_zero_ws)
        flags_ws = detect_quality_flags(df_ws)
        self.assertIn("zero_wheel_speed", flags_ws)
        self.assertNotIn("zero_steering", flags_ws)

        csv_zero_st = self.temp_dir / "zero_st.csv"
        _make_synthetic_can_csv(csv_zero_st, ws_fl=[12.0] * 10, steer=[0.0] * 10, n_rows=10)
        df_st = parse_can_csv(csv_zero_st)
        flags_st = detect_quality_flags(df_st)
        self.assertIn("zero_steering", flags_st)
        self.assertNotIn("zero_wheel_speed", flags_st)

    def test_deterministic_parquet(self):
        """Verify writing the same dataframe twice produces byte-identical Parquet files."""
        csv_file = self.temp_dir / "det_run.csv"
        _make_synthetic_can_csv(csv_file, n_rows=15)
        df = parse_can_csv(csv_file)

        out1 = self.temp_dir / "out1"
        out2 = self.temp_dir / "out2"
        p1 = write_parquet(df, "V-Test", out1)
        p2 = write_parquet(df, "V-Test", out2)

        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
        self.assertEqual(h1, h2, f"Parquet serialization non-deterministic: {h1} != {h2}")


if __name__ == "__main__":
    unittest.main()
