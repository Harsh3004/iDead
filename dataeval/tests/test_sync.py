"""Unit tests for paired CAN and Smartphone time synchronization and grid alignment."""

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from dataeval.harness.pairing import PairEntry, discover_pairs
from dataeval.harness.sync import (
    AlignedPairResult,
    OffsetResult,
    _circular_interp,
    _step_interp,
    align_pair,
    estimate_offset,
    write_paired_parquet,
)


def _make_synthetic_pair(
    n_samples: int = 200,
    offset_samples: int = 25,  # 2.5 seconds at 10 Hz
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic CAN and Phone DataFrames with an injected time offset."""
    # Distinct velocity waveform: acceleration and deceleration
    t = np.linspace(0, 4 * np.pi, n_samples)
    speed_profile = 40.0 + 30.0 * np.sin(t) + 10.0 * np.cos(2 * t)

    can_rows = n_samples
    phone_rows = n_samples

    # CAN speed is immediate
    can_speed = speed_profile.copy()
    # Phone speed is delayed by offset_samples
    phone_speed = np.zeros(phone_rows)
    if offset_samples >= 0:
        phone_speed[offset_samples:] = speed_profile[: phone_rows - offset_samples]
    else:
        phone_speed[: phone_rows + offset_samples] = speed_profile[-offset_samples:]

    # CAN DataFrame
    can_df = pd.DataFrame(
        {
            "gps_num_satellites": np.array([12] * can_rows, dtype=np.int32),
            "timestamp_s": np.arange(can_rows) * 0.1 + 30000.0,
            "latitude_deg": 52.4 + np.linspace(0, 0.01, can_rows),
            "longitude_deg": -1.5 + np.linspace(0, 0.01, can_rows),
            "gps_velocity_kmh": can_speed.astype(np.float32),
            "gps_heading_deg": np.linspace(0, 180, can_rows).astype(np.float32),
            "height_m": np.array([120.0] * can_rows, dtype=np.float32),
            "gps_vertical_velocity_kmh": np.zeros(can_rows, dtype=np.float32),
            "sample_period_s": np.array([0.1] * can_rows, dtype=np.float32),
            "steering_angle_deg": np.zeros(can_rows, dtype=np.float32),
            "wheel_speed_fl_rad_s": (can_speed / 3.6 / 0.3).astype(np.float32),
            "wheel_speed_fr_rad_s": (can_speed / 3.6 / 0.3).astype(np.float32),
            "wheel_speed_rl_rad_s": (can_speed / 3.6 / 0.3).astype(np.float32),
            "wheel_speed_rr_rad_s": (can_speed / 3.6 / 0.3).astype(np.float32),
            "yaw_rate_deg_s": np.zeros(can_rows, dtype=np.float32),
            "indicated_vehicle_speed_kmh": can_speed.astype(np.float32),
            "indicated_longitudinal_accel_g": np.zeros(can_rows, dtype=np.float32),
            "indicated_lateral_accel_g": np.zeros(can_rows, dtype=np.float32),
            "handbrake": np.array([0] * can_rows, dtype=np.int8),
            "gear_requested": np.array([2] * (can_rows // 2) + [3] * (can_rows - can_rows // 2), dtype=np.int8),
            "gear_actual": np.array([2] * (can_rows // 2) + [3] * (can_rows - can_rows // 2), dtype=np.int8),
            "engine_speed_rpm": (can_speed * 40.0).astype(np.float32),
            "coolant_temp_c": np.array([85.0] * can_rows, dtype=np.float32),
            "clutch_position": np.zeros(can_rows, dtype=np.int8),
            "brake_pressure_psi": np.zeros(can_rows, dtype=np.float32),
            "brake_position": np.zeros(can_rows, dtype=np.int8),
            "battery_voltage_v": np.array([13.8] * can_rows, dtype=np.float32),
            "air_temp_c": np.array([18.0] * can_rows, dtype=np.float32),
            "accelerator_pedal_pct": np.array([25.0] * can_rows, dtype=np.float32),
        }
    )

    # Phone DataFrame
    phone_df = pd.DataFrame(
        {
            "timestamp_ms": (np.arange(phone_rows) * 100).astype(np.int64),
            "timestamp_s": np.arange(phone_rows) * 0.1,
            "date_utc": pd.Series([f"2019-09-08 10:00:{i:02d}:000" for i in range(phone_rows)]),
            "latitude_deg": 52.4 + np.linspace(0, 0.01, phone_rows),
            "longitude_deg": -1.5 + np.linspace(0, 0.01, phone_rows),
            "altitude_m": np.array([120.0] * phone_rows, dtype=np.float32),
            "gps_speed_raw": phone_speed.astype(np.float32),
            "gps_speed_ms": (phone_speed / 3.6).astype(np.float32),
            "gps_speed_kmh": phone_speed.astype(np.float32),
            "gps_accuracy_m": np.array([3.0] * phone_rows, dtype=np.float32),
            "gps_bearing_deg": np.linspace(0, 180, phone_rows).astype(np.float32),
            "gps_satellites_in_range": pd.Series([18] * phone_rows, dtype="Int16"),
            "accel_x_m_s2": np.zeros(phone_rows, dtype=np.float32),
            "accel_y_m_s2": np.zeros(phone_rows, dtype=np.float32),
            "accel_z_m_s2": np.array([9.8] * phone_rows, dtype=np.float32),
            "gravity_x_m_s2": np.zeros(phone_rows, dtype=np.float32),
            "gravity_y_m_s2": np.zeros(phone_rows, dtype=np.float32),
            "gravity_z_m_s2": np.array([9.8] * phone_rows, dtype=np.float32),
            "gyro_x_rad_s": np.zeros(phone_rows, dtype=np.float32),
            "gyro_y_rad_s": np.zeros(phone_rows, dtype=np.float32),
            "gyro_z_rad_s": np.zeros(phone_rows, dtype=np.float32),
            "mag_x_uT": np.array([-15.0] * phone_rows, dtype=np.float32),
            "mag_y_uT": np.array([2.5] * phone_rows, dtype=np.float32),
            "mag_z_uT": np.array([-38.0] * phone_rows, dtype=np.float32),
            "orientation_azimuth_deg": np.linspace(0, 180, phone_rows).astype(np.float32),
            "orientation_pitch_deg": np.zeros(phone_rows, dtype=np.float32),
            "orientation_roll_deg": np.zeros(phone_rows, dtype=np.float32),
        }
    )

    return can_df, phone_df


class TestSync(unittest.TestCase):
    """Test suite for CAN-Smartphone offset estimation and grid alignment."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_offset_recovery_synthetic(self):
        """Verify cross-correlation recovers injected 2.5s time offset with high confidence."""
        can_df, phone_df = _make_synthetic_pair(n_samples=300, offset_samples=25)
        res: OffsetResult = estimate_offset(can_df, phone_df, max_lag_s=10.0)

        # Injected offset: 25 samples * 0.1s = 2.5s
        self.assertAlmostEqual(res.estimated_offset_s, 2.5, delta=0.1)
        self.assertGreater(res.offset_confidence, 0.85)
        self.assertNotIn("low_confidence_offset", res.quality_flags)

    def test_intersection_no_extrapolation(self):
        """Verify aligned pair produces exact intersection without extrapolating beyond recorded bounds."""
        can_df, phone_df = _make_synthetic_pair(n_samples=200, offset_samples=20)  # 2.0s offset
        res = align_pair(can_df, phone_df, offset_s=2.0)

        # CAN covers [0.0, 19.9s], Phone_on_CAN covers [-2.0s, 17.9s]
        # Intersection must be [0.0, 17.9s] = 180 samples
        expected_rows = 180
        self.assertEqual(res.aligned_n_rows, expected_rows)
        self.assertAlmostEqual(res.aligned_duration_s, 17.9, delta=0.05)
        self.assertLess(res.overlap_pct_of_can, 100.0)
        self.assertLess(res.overlap_pct_of_phone, 100.0)

        # Check column count: 1 shared timestamp_s + 29 CAN + 27 Phone = 57 columns
        self.assertEqual(len(res.aligned_df.columns), 57)
        self.assertEqual(res.aligned_df.columns[0], "timestamp_s")
        self.assertTrue(all(c.startswith("can_") or c.startswith("phone_") or c == "timestamp_s" for c in res.aligned_df.columns))

    def test_non_interpolatable_columns(self):
        """Verify categorical/flag columns are step-interpolated without fractional values."""
        can_df, phone_df = _make_synthetic_pair(n_samples=100, offset_samples=0)
        res = align_pair(can_df, phone_df, offset_s=0.0)

        df = res.aligned_df
        # Gear requested must only contain integers 2 and 3, never fractional e.g. 2.5
        gears = df["can_gear_requested"].unique()
        self.assertTrue(set(gears).issubset({2, 3}))

        # Handbrake must only be 0
        self.assertTrue((df["can_handbrake"] == 0).all())

        # Phone satellites must only be 18 or null
        sats = df["phone_gps_satellites_in_range"].dropna().unique()
        self.assertTrue(set(sats).issubset({18}))

    def test_circular_heading_interpolation(self):
        """Verify circular angles wrap across 360 -> 0 through 0 deg, not through 180 deg."""
        times = np.array([0.0, 1.0])
        angles = np.array([355.0, 5.0])
        target_t = np.array([0.5])

        interp = _circular_interp(times, angles, target_t)
        # Midpoint between 355 and 5 through 0 is 0 deg (or 360 deg)
        self.assertTrue(np.isclose(interp[0], 0.0, atol=1.0) or np.isclose(interp[0], 360.0, atol=1.0))

    def test_stationary_run_handling(self):
        """Verify stationary runs (zero velocity variance) yield zero offset and stationary flag."""
        can_df, phone_df = _make_synthetic_pair(n_samples=100, offset_samples=0)
        can_df["gps_velocity_kmh"] = 0.0
        phone_df["gps_speed_kmh"] = 0.0

        res = estimate_offset(can_df, phone_df)
        self.assertEqual(res.estimated_offset_s, 0.0)
        self.assertEqual(res.offset_confidence, 0.0)
        self.assertIn("stationary_run", res.quality_flags)

    def test_missing_side_discovery(self):
        """Verify discover_pairs correctly tags runs with a missing counterpart."""
        can_manifest_file = self.temp_dir / "can_manifest.csv"
        phone_manifest_file = self.temp_dir / "phone_manifest.csv"

        # Create dummy manifests where CAN has S1, but Phone does not
        can_df = pd.DataFrame(
            [
                {
                    "run_id": "V-S1",
                    "canonical_source_path": "Synchronised/V-S1.csv",
                    "duplicate_source_paths": "",
                    "parse_status": "ok",
                    "quality_flags": "none",
                }
            ]
        )
        phone_df = pd.DataFrame(
            [
                {
                    "run_id": "S-S2",
                    "canonical_source_path": "Synchronised/S-S2.csv",
                    "duplicate_source_paths": "",
                    "parse_status": "ok",
                    "quality_flags": "none",
                }
            ]
        )
        can_df.to_csv(can_manifest_file, index=False)
        phone_df.to_csv(phone_manifest_file, index=False)

        pairs = discover_pairs(can_manifest_file, phone_manifest_file, processed_root=self.temp_dir)
        self.assertEqual(len(pairs), 2)
        by_pair = {p.pair_id: p for p in pairs}
        self.assertEqual(by_pair["pair_S1"].status, "missing_phone")
        self.assertEqual(by_pair["pair_S2"].status, "missing_can")

    def test_deterministic_parquet(self):
        """Verify writing the same aligned dataframe twice produces byte-identical Parquet files."""
        can_df, phone_df = _make_synthetic_pair(n_samples=50, offset_samples=0)
        aligned_res = align_pair(can_df, phone_df, offset_s=0.0)

        out1 = self.temp_dir / "out1"
        out2 = self.temp_dir / "out2"

        p1 = write_paired_parquet(aligned_res.aligned_df, "pair_test", out1)
        p2 = write_paired_parquet(aligned_res.aligned_df, "pair_test", out2)

        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
        self.assertEqual(h1, h2, f"Non-deterministic Parquet serialization: {h1} != {h2}")


if __name__ == "__main__":
    unittest.main()
