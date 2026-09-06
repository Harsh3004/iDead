"""Unit tests for smartphone CSV dataset ingestion and parsing."""

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from dataeval.ingest.smartphone import (
    CANONICAL_COLUMN_ORDER,
    _parse_satellite_count,
    detect_phone_quality_flags,
    detect_schema_variant,
    discover_phone_runs,
    normalize_phone_run_id,
    parse_phone_csv,
    write_parquet,
)


def _make_synthetic_phone_csv(
    path: Path,
    n_rows: int = 20,
    variant: str = "standard_24",
    **kwargs,
) -> Path:
    """Helper to generate synthetic smartphone CSV fixtures for all schema variants."""
    t_ms = kwargs.get("t_ms", [i * 100 for i in range(n_rows)])
    lats = kwargs.get("lats", [52.4 + i * 0.0001 for i in range(n_rows)])
    lons = kwargs.get("lons", [-1.5 + i * 0.0001 for i in range(n_rows)])
    speeds = kwargs.get("speeds", [10.0 for _ in range(n_rows)])
    sats = kwargs.get("sats", ["18 / 19" for _ in range(n_rows)])

    path.parent.mkdir(parents=True, exist_ok=True)

    if variant == "corrupted_delimiter_25":
        # Trailing comma in header, stray empty field after GPS ORIENTATION
        header = (
            "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh), "
            "GPS ACCURACY (m), GPS ORIENTATION (°),GPS SATELLITES IN RANGE, TIME SINCE START (ms), "
            "DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X (m/s²) , ACCELEROMETER Y (m/s²), "
            "ACCELEROMETER Z (m/s²), GRAVITY X (m/s²), GRAVITY Y (m/s²), GRAVITY Z (m/s²), "
            "GYROSCOPE Yaw (rad/s), GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s), "
            "MAGNETIC FIELD X (μT), MAGNETIC FIELD Y (μT), MAGNETIC FIELD Z (μT), "
            "ORIENTATION (Yaw) (°), ORIENTATION (Pitch) (°), ORIENTATION (Roll ) (°),\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for i in range(n_rows):
                row = (
                    f"{lats[i]},{lons[i]},145.0,{speeds[i]},3.0,180.0,,{sats[i]},{t_ms[i]},"
                    f"2019-09-08 10:00:00:000,0.1,-0.2,9.8,0.0,0.0,9.8,0.01,-0.02,0.005,"
                    f"-15.0,2.5,-38.0,185.0,0.5,-0.8\n"
                )
                f.write(row)
        return path

    if variant == "truncated_18":
        header = (
            "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh), "
            "GPS ACCURACY (m), GPS ORIENTATION (°),SATELLITES IN RANGE, TIME SINCE START (ms), "
            "DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X (m/s²) , ACCELEROMETER Y (m/s²), "
            "ACCELEROMETER Z (m/s²), GRAVITY X (m/s²), GRAVITY Y (m/s²), GRAVITY Z (m/s²), "
            "GYROSCOPE Yaw (rad/s), GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s)\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for i in range(n_rows):
                row = (
                    f"{lats[i]},{lons[i]},145.0,{speeds[i]},3.0,180.0,{sats[i]},{t_ms[i]},"
                    f"2019-09-08 10:00:00:000,0.1,-0.2,9.8,0.0,0.0,9.8,0.01,-0.02,0.005\n"
                )
                f.write(row)
        return path

    if variant == "typo_header_24":
        date_header = "DATE (YYYY-MO-DD HH-MI-SS_SSS"
    else:
        date_header = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"

    header = (
        f"GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh), "
        f"GPS ACCURACY (m), GPS ORIENTATION (°),GPS SATELLITES IN RANGE, TIME SINCE START (ms), "
        f"{date_header}, ACCELEROMETER X (m/s²) , ACCELEROMETER Y (m/s²), "
        f"ACCELEROMETER Z (m/s²), GRAVITY X (m/s²), GRAVITY Y (m/s²), GRAVITY Z (m/s²), "
        f"GYROSCOPE X (rad/s), GYROSCOPE Y (rad/s), GYROSCOPE Z (rad/s), "
        f"MAGNETIC FIELD X (μT), MAGNETIC FIELD Y (μT), MAGNETIC FIELD Z (μT), "
        f"ORIENTATION (Azimuth) (°), ORIENTATION (Pitch) (°), ORIENTATION (Roll ) (°)\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(n_rows):
            row = (
                f"{lats[i]},{lons[i]},145.0,{speeds[i]},3.0,180.0,{sats[i]},{t_ms[i]},"
                f"2019-09-08 10:00:00:000,0.1,-0.2,9.8,0.0,0.0,9.8,0.01,-0.02,0.005,"
                f"-15.0,2.5,-38.0,185.0,0.5,-0.8\n"
            )
            f.write(row)
    return path


class TestSmartphoneIngest(unittest.TestCase):
    """Test suite for smartphone CSV parsing, variant handling, and Parquet serialization."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_variants_produce_identical_schema(self):
        """Verify that standard 24, typo 24, truncated 18, and corrupted 25 all yield the exact 27 canonical columns."""
        variants = ["standard_24", "typo_header_24", "truncated_18", "corrupted_delimiter_25"]
        for var in variants:
            p = self.temp_dir / f"test_{var}.csv"
            _make_synthetic_phone_csv(p, variant=var)
            detected = detect_schema_variant(p)
            self.assertEqual(detected, var, f"Variant mismatch for {var}: detected {detected}")

            df, dt = parse_phone_csv(p, variant=detected)
            self.assertEqual(
                list(df.columns),
                CANONICAL_COLUMN_ORDER,
                f"Column mismatch for variant {var}",
            )
            self.assertEqual(len(df.columns), 27)
            self.assertEqual(len(df), 20)

    def test_truncated_18_produces_real_nulls(self):
        """Verify French 18-col variant populates magnetometer and orientation with real nulls, not zeros."""
        p = self.temp_dir / "test_french.csv"
        _make_synthetic_phone_csv(p, variant="truncated_18")
        df, _ = parse_phone_csv(p)

        null_cols = [
            "mag_x_uT",
            "mag_y_uT",
            "mag_z_uT",
            "orientation_azimuth_deg",
            "orientation_pitch_deg",
            "orientation_roll_deg",
        ]
        for c in null_cols:
            self.assertTrue(df[c].isna().all(), f"Column {c} should be all nulls")
            self.assertFalse((df[c] == 0.0).any(), f"Column {c} must not be zero-filled")

        # Accel, gyro, gravity must be intact
        self.assertFalse(df["accel_z_m_s2"].isna().any())
        self.assertFalse(df["gyro_x_rad_s"].isna().any())

    def test_delimiter_shifted_repair(self):
        """Verify corrupted delimiter variant (S-A4) is cleanly repaired without shifting columns."""
        p = self.temp_dir / "S-A4.csv"
        _make_synthetic_phone_csv(p, variant="corrupted_delimiter_25")
        df, _ = parse_phone_csv(p)

        # Confirm accel_x is a float near 0.1, not a date string
        self.assertEqual(df["accel_x_m_s2"].dtype, np.float32)
        self.assertTrue(np.isclose(df["accel_x_m_s2"].iloc[0], 0.1))

        # Confirm orientation_roll_deg is -0.8
        self.assertTrue(np.isclose(df["orientation_roll_deg"].iloc[0], -0.8))

        # Quality flag check
        rate_class, flags = detect_phone_quality_flags(df, "corrupted_delimiter_25", 0.1, "s-a4")
        self.assertIn("delimiter_repaired", flags)

    def test_speed_unit_conversions(self):
        """Verify gps_speed_ms and gps_speed_kmh for both m/s standard runs and km/h special runs."""
        # 1. Standard run where raw is in m/s (e.g. 10.0 m/s)
        p_std = self.temp_dir / "S-S1.csv"
        _make_synthetic_phone_csv(p_std, speeds=[10.0] * 5, n_rows=5)
        df_std, _ = parse_phone_csv(p_std)
        self.assertTrue(np.isclose(df_std["gps_speed_ms"].iloc[0], 10.0))
        self.assertTrue(np.isclose(df_std["gps_speed_kmh"].iloc[0], 36.0))

        # 2. S-A2 where raw was recorded in km/h (e.g. 72.0 km/h)
        p_sa2 = self.temp_dir / "S-A2.csv"
        _make_synthetic_phone_csv(p_sa2, speeds=[72.0] * 5, n_rows=5)
        df_sa2, _ = parse_phone_csv(p_sa2)
        self.assertTrue(np.isclose(df_sa2["gps_speed_kmh"].iloc[0], 72.0))
        self.assertTrue(np.isclose(df_sa2["gps_speed_ms"].iloc[0], 20.0))

    def test_mangled_satellite_parsing(self):
        """Verify tolerant satellite parsing extracts counts from '18 / 19' and nulls from 'Aug-20'."""
        self.assertEqual(_parse_satellite_count("18 / 19"), 18)
        self.assertEqual(_parse_satellite_count("7 / 20"), 7)
        self.assertEqual(_parse_satellite_count("29 / 29"), 29)
        self.assertIsNone(_parse_satellite_count("Aug-20"))
        self.assertIsNone(_parse_satellite_count("Sep-19"))
        self.assertIsNone(_parse_satellite_count(None))
        self.assertIsNone(_parse_satellite_count(np.nan))

    def test_clock_reset_flagging(self):
        """Verify negative dt (clock reset) is detected and tagged without altering rows."""
        p = self.temp_dir / "S-M.csv"
        times = [0, 100, 200, 150, 250]  # row 3 has dt < 0
        _make_synthetic_phone_csv(p, t_ms=times, n_rows=len(times))
        df, dt = parse_phone_csv(p)

        self.assertEqual(len(df), len(times))
        rate_class, flags = detect_phone_quality_flags(df, "standard_24", dt, "s-m")
        self.assertIn("clock_reset", flags)

    def test_dedup_prefers_categorised(self):
        """Verify discover_phone_runs prioritizes Categorised copies over Uncategorised copies."""
        raw_root = self.temp_dir / "raw_dataset"

        cat_path = (
            raw_root
            / "Synchronised V abd S datasets"
            / "Categorised IOVNB Dataset"
            / "S (Driver A)"
            / "S1"
            / "S-S1.csv"
        )
        uncat_path = (
            raw_root
            / "Synchronised V abd S datasets"
            / "Uncategorised IOVNB Dataset"
            / "S-Dataset"
            / "S-S1.csv"
        )

        _make_synthetic_phone_csv(cat_path)
        _make_synthetic_phone_csv(uncat_path)

        entries = discover_phone_runs(raw_root)
        self.assertEqual(len(entries), 1)

        entry = entries[0]
        self.assertEqual(entry.run_id, "S-S1")
        self.assertIn("Categorised", entry.canonical_source_path)
        self.assertIn("Uncategorised", entry.duplicate_source_paths)

    def test_deterministic_parquet(self):
        """Verify writing the same dataframe twice produces byte-identical Parquet files."""
        csv_file = self.temp_dir / "det_run.csv"
        _make_synthetic_phone_csv(csv_file, n_rows=15)
        df, _ = parse_phone_csv(csv_file)

        out1 = self.temp_dir / "out1"
        out2 = self.temp_dir / "out2"
        p1 = write_parquet(df, "S-Test", out1)
        p2 = write_parquet(df, "S-Test", out2)

        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
        self.assertEqual(h1, h2, f"Parquet serialization non-deterministic: {h1} != {h2}")


if __name__ == "__main__":
    unittest.main()
