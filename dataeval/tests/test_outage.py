"""Unit tests for GNSS outage simulator harness."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataeval.harness.outage import (
    CAN_GNSS_COLUMNS,
    DEFAULT_EDGE_MARGIN_S,
    PHONE_GNSS_COLUMNS,
    OutageSkip,
    OutageWindow,
    OutageWindowList,
    apply_outage,
    find_real_gaps,
    ground_truth_trajectory,
    select_outage_windows,
)


class TestOutageSimulator(unittest.TestCase):
    """Test suite for synthetic GNSS outage window selection, masking, and evaluation."""

    def setUp(self):
        # Build synthetic 10 Hz paired DataFrame (duration 200s, 2000 rows)
        n_rows = 2000
        t = np.linspace(0.0, 199.9, n_rows)

        self.paired_df = pd.DataFrame({
            "timestamp_s": t,
            # CAN GNSS
            "can_latitude_deg": 52.4 + 1e-4 * np.sin(t / 10.0),
            "can_longitude_deg": -1.5 + 1e-4 * np.cos(t / 10.0),
            "can_gps_velocity_kmh": np.full(n_rows, 50.0, dtype=np.float32),
            "can_gps_heading_deg": np.full(n_rows, 90.0, dtype=np.float32),
            "can_gps_vertical_velocity_kmh": np.zeros(n_rows, dtype=np.float32),
            "can_gps_num_satellites": np.full(n_rows, 12, dtype=np.int32),
            "can_height_m": np.full(n_rows, 105.0, dtype=np.float32),
            # CAN Non-GNSS (inertial, odometry, steering)
            "can_steering_angle_deg": np.full(n_rows, 15.0, dtype=np.float32),
            "can_wheel_speed_fl_rad_s": np.full(n_rows, 42.0, dtype=np.float32),
            "can_wheel_speed_fr_rad_s": np.full(n_rows, 42.1, dtype=np.float32),
            "can_yaw_rate_deg_s": np.full(n_rows, 1.2, dtype=np.float32),
            "can_indicated_vehicle_speed_kmh": np.full(n_rows, 50.5, dtype=np.float32),
            # Phone GNSS
            "phone_latitude_deg": 52.4 + 1e-4 * np.sin(t / 10.0) + 1e-6,
            "phone_longitude_deg": -1.5 + 1e-4 * np.cos(t / 10.0) + 1e-6,
            "phone_altitude_m": np.full(n_rows, 105.2, dtype=np.float32),
            "phone_gps_speed_ms": np.full(n_rows, 13.88, dtype=np.float32),
            "phone_gps_speed_kmh": np.full(n_rows, 50.0, dtype=np.float32),
            "phone_gps_speed_raw": np.full(n_rows, 13.88, dtype=np.float32),
            "phone_gps_accuracy_m": np.full(n_rows, 3.5, dtype=np.float32),
            "phone_gps_bearing_deg": np.full(n_rows, 90.1, dtype=np.float32),
            "phone_gps_satellites_in_range": pd.Series([10] * n_rows, dtype="Int16"),
            # Phone Non-GNSS (IMU, orientation)
            "phone_accel_x_m_s2": np.full(n_rows, 0.5, dtype=np.float32),
            "phone_accel_y_m_s2": np.full(n_rows, 0.2, dtype=np.float32),
            "phone_accel_z_m_s2": np.full(n_rows, 9.81, dtype=np.float32),
            "phone_gyro_x_rad_s": np.full(n_rows, 0.01, dtype=np.float32),
            "phone_gyro_y_rad_s": np.full(n_rows, 0.02, dtype=np.float32),
            "phone_gyro_z_rad_s": np.full(n_rows, 0.03, dtype=np.float32),
            "phone_mag_x_uT": np.full(n_rows, 20.0, dtype=np.float32),
            "phone_mag_y_uT": np.full(n_rows, -15.0, dtype=np.float32),
            "phone_mag_z_uT": np.full(n_rows, 45.0, dtype=np.float32),
        })

    def test_window_bounds_and_no_extrapolation(self):
        """Window start and end fall strictly within [min_ts + margin, max_ts - margin]."""
        margin = 10.0
        durations = [10.0, 30.0, 60.0]
        windows = select_outage_windows(
            self.paired_df,
            run_id="test_run",
            durations=durations,
            side="phone",
            seed=42,
            edge_margin_s=margin,
        )

        self.assertEqual(len(windows), 3)
        min_ts = self.paired_df["timestamp_s"].min()
        max_ts = self.paired_df["timestamp_s"].max()

        for w in windows:
            self.assertGreaterEqual(w.start_s, min_ts + margin)
            self.assertLessEqual(w.end_s, max_ts - margin)
            self.assertAlmostEqual(w.end_s - w.start_s, w.duration_s, places=2)
            self.assertFalse(w.overlaps_real_gap)

    def test_null_masking_and_non_gnss_invariance(self):
        """Target GNSS columns inside window are genuine nulls; inertial/chassis columns are untouched."""
        window = OutageWindow(
            run_id="test_run",
            source_side="phone",
            start_s=50.0,
            duration_s=30.0,
            end_s=80.0,
        )
        masked_df = apply_outage(self.paired_df, window)

        # 1. gnss_outage_active column
        self.assertIn("gnss_outage_active", masked_df.columns)
        expected_active = (self.paired_df["timestamp_s"] >= 50.0) & (self.paired_df["timestamp_s"] < 80.0)
        np.testing.assert_array_equal(masked_df["gnss_outage_active"].values, expected_active.values)

        # 2. Inside window: Phone GNSS channels must be genuine null/NaN (never 0.0)
        active_slice = masked_df[masked_df["gnss_outage_active"]]
        self.assertGreater(len(active_slice), 0)

        for col in [
            "phone_latitude_deg",
            "phone_longitude_deg",
            "phone_altitude_m",
            "phone_gps_speed_ms",
            "phone_gps_speed_kmh",
            "phone_gps_speed_raw",
            "phone_gps_accuracy_m",
            "phone_gps_bearing_deg",
            "phone_gps_satellites_in_range",
        ]:
            self.assertTrue(
                active_slice[col].isna().all(),
                f"Column {col} should be 100% null inside outage window, but had non-nulls",
            )

        # 3. Outside window: Phone GNSS channels must match original exactly
        inactive_slice = masked_df[~masked_df["gnss_outage_active"]]
        for col in [
            "phone_latitude_deg",
            "phone_longitude_deg",
            "phone_altitude_m",
            "phone_gps_speed_ms",
        ]:
            np.testing.assert_array_equal(
                inactive_slice[col].values,
                self.paired_df.loc[~expected_active, col].values,
            )

        # 4. Non-GNSS channels (IMU, steering, wheel speed, CAN GNSS) MUST be completely untouched!
        for col in [
            "can_latitude_deg",
            "can_longitude_deg",
            "can_gps_velocity_kmh",
            "can_steering_angle_deg",
            "can_wheel_speed_fl_rad_s",
            "phone_accel_x_m_s2",
            "phone_accel_y_m_s2",
            "phone_accel_z_m_s2",
            "phone_gyro_z_rad_s",
            "phone_mag_x_uT",
        ]:
            np.testing.assert_array_equal(
                masked_df[col].values,
                self.paired_df[col].values,
                err_msg=f"Non-GNSS column {col} was modified by apply_outage!",
            )

    def test_can_side_masking(self):
        """When source_side='can', CAN GNSS columns become null and Phone channels remain untouched."""
        window = OutageWindow(
            run_id="test_run",
            source_side="can",
            start_s=40.0,
            duration_s=20.0,
            end_s=60.0,
        )
        masked_df = apply_outage(self.paired_df, window)
        active_slice = masked_df[masked_df["gnss_outage_active"]]

        # CAN GNSS null
        for col in [
            "can_latitude_deg",
            "can_longitude_deg",
            "can_gps_velocity_kmh",
            "can_gps_heading_deg",
            "can_height_m",
            "can_gps_num_satellites",
        ]:
            self.assertTrue(active_slice[col].isna().all(), f"CAN {col} should be null")

        # Phone GNSS untouched
        for col in ["phone_latitude_deg", "phone_longitude_deg", "phone_gps_speed_ms"]:
            np.testing.assert_array_equal(
                masked_df[col].values,
                self.paired_df[col].values,
            )

    def test_run_shorter_than_duration_produces_skip(self):
        """Runs shorter than outage duration produce documented OutageSkip, not a clipped window."""
        short_df = pd.DataFrame({
            "timestamp_s": np.linspace(0.0, 45.0, 451),
            "phone_latitude_deg": np.full(451, 52.4),
        })

        # With 10s edge margin, a 45s run requires 45s >= dur + 20s -> max duration is 25s
        res, skips = select_outage_windows(
            short_df,
            run_id="short_run",
            durations=[10.0, 30.0, 60.0, 180.0],
            side="phone",
            edge_margin_s=10.0,
            return_skipped=True,
        )

        # 10s fits (10 + 20 = 30 <= 45)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].duration_s, 10.0)

        # 30s, 60s, 180s must be skipped
        self.assertEqual(len(skips), 3)
        skip_durs = [s.duration_s for s in skips]
        self.assertEqual(skip_durs, [30.0, 60.0, 180.0])
        for s in skips:
            self.assertIn("run_too_short", s.reason)
            self.assertEqual(s.run_id, "short_run")

    def test_real_gap_avoidance_by_default_and_opt_in_override(self):
        """Candidate windows avoid real recording gaps by default, and opt-in allows/flags them."""
        # Create run with an authentic recording gap from t=60.0 to t=120.0 (60s gap)
        t_part1 = np.linspace(0.0, 60.0, 601)
        t_part2 = np.linspace(120.0, 200.0, 801)
        t_all = np.concatenate([t_part1, t_part2])

        gap_df = pd.DataFrame({
            "timestamp_s": t_all,
            "phone_latitude_deg": np.full(len(t_all), 52.4),
            "phone_longitude_deg": np.full(len(t_all), -1.5),
        })

        # Check gap detection
        gaps = find_real_gaps(gap_df, gap_threshold_s=2.0)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0][0], 60.0, places=1)
        self.assertAlmostEqual(gaps[0][1], 120.0, places=1)

        # 1. Default: avoid_real_gaps=True
        windows_avoid = select_outage_windows(
            gap_df,
            run_id="gap_run",
            durations=[20.0],
            side="phone",
            n_per_duration=1,
            seed=42,
            avoid_real_gaps=True,
            edge_margin_s=10.0,
        )
        self.assertEqual(len(windows_avoid), 1)
        w = windows_avoid[0]
        self.assertFalse(w.overlaps_real_gap)
        # Window must not overlap [60.0, 120.0]
        self.assertTrue(w.end_s <= 60.0 or w.start_s >= 120.0)

        # 2. Opt-in: avoid_real_gaps=False
        # For a 70s duration, it CANNOT fit in segment 1 (60s) or segment 2 (80s - 20s margin = 60s)
        # without crossing the gap.
        # With avoid_real_gaps=True, 70s should skip.
        win_avoid_70, skips_avoid_70 = select_outage_windows(
            gap_df,
            run_id="gap_run",
            durations=[70.0],
            side="phone",
            avoid_real_gaps=True,
            edge_margin_s=10.0,
            return_skipped=True,
        )
        self.assertEqual(len(win_avoid_70), 0)
        self.assertEqual(len(skips_avoid_70), 1)
        self.assertIn("overlaps_real_gap", skips_avoid_70[0].reason)

        # With avoid_real_gaps=False, 70s should place and flag overlaps_real_gap=True
        win_allow_70 = select_outage_windows(
            gap_df,
            run_id="gap_run",
            durations=[70.0],
            side="phone",
            avoid_real_gaps=False,
            edge_margin_s=10.0,
        )
        self.assertEqual(len(win_allow_70), 1)
        self.assertTrue(win_allow_70[0].overlaps_real_gap)

    def test_ground_truth_trajectory_paired_vs_unpaired(self):
        """ground_truth_trajectory returns other-stream data for paired runs and None for unpaired runs."""
        window = OutageWindow(
            run_id="pair_run",
            source_side="phone",
            start_s=30.0,
            duration_s=20.0,
            end_s=50.0,
        )

        # 1. Paired run: returns CAN ground truth
        gt = ground_truth_trajectory(self.paired_df, window)
        self.assertIsNotNone(gt)
        self.assertIsInstance(gt, pd.DataFrame)
        self.assertIn("latitude_deg", gt.columns)
        self.assertIn("longitude_deg", gt.columns)
        self.assertIn("speed_ms", gt.columns)
        self.assertIn("heading_deg", gt.columns)
        self.assertAlmostEqual(gt["timestamp_s"].min(), 30.0, places=1)
        self.assertAlmostEqual(gt["timestamp_s"].max(), 49.9, places=1)
        # Should match unmasked CAN latitude
        expected_lat = self.paired_df.loc[
            (self.paired_df["timestamp_s"] >= 30.0) & (self.paired_df["timestamp_s"] < 50.0),
            "can_latitude_deg",
        ].values
        np.testing.assert_array_equal(gt["latitude_deg"].values, expected_lat)

        # 2. Unpaired run (phone only): CAN columns absent
        phone_only_df = self.paired_df[[
            c for c in self.paired_df.columns if not c.startswith("can_")
        ]].copy()
        gt_unpaired = ground_truth_trajectory(phone_only_df, window)
        self.assertIsNone(gt_unpaired)

        # 3. Source side "both": returns None
        both_window = OutageWindow(
            run_id="pair_run",
            source_side="both",
            start_s=30.0,
            duration_s=20.0,
            end_s=50.0,
        )
        gt_both = ground_truth_trajectory(self.paired_df, both_window)
        self.assertIsNone(gt_both)

    def test_determinism_same_seed(self):
        """Identical seed produces exact same windows across multiple invocations."""
        durations = [10.0, 30.0, 60.0, 120.0]
        run1 = select_outage_windows(self.paired_df, "run_det", durations=durations, seed=12345)
        run2 = select_outage_windows(self.paired_df, "run_det", durations=durations, seed=12345)

        self.assertEqual(len(run1), len(run2))
        for w1, w2 in zip(run1, run2):
            self.assertEqual(w1.start_s, w2.start_s)
            self.assertEqual(w1.end_s, w2.end_s)
            self.assertEqual(w1.duration_s, w2.duration_s)
            self.assertEqual(w1.overlaps_real_gap, w2.overlaps_real_gap)

        # Different seed produces different placement
        run3 = select_outage_windows(self.paired_df, "run_det", durations=durations, seed=99999)
        diffs = [w1.start_s != w3.start_s for w1, w3 in zip(run1, run3)]
        self.assertTrue(any(diffs))

    def test_multiple_windows_per_duration(self):
        """When n_per_duration > 1, chosen windows for the same duration do not overlap."""
        windows = select_outage_windows(
            self.paired_df,
            run_id="test_run",
            durations=[20.0],
            n_per_duration=3,
            seed=42,
        )
        self.assertEqual(len(windows), 3)
        # Check pairwise non-overlap
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                w_i = windows[i]
                w_j = windows[j]
                overlap = max(w_i.start_s, w_j.start_s) < min(w_i.end_s, w_j.end_s)
                self.assertFalse(
                    overlap,
                    f"Windows {i} ([{w_i.start_s}, {w_i.end_s}]) and {j} ([{w_j.start_s}, {w_j.end_s}]) overlap!",
                )


if __name__ == "__main__":
    unittest.main()
