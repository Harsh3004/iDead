"""Unit tests for naive constant-velocity / heading dead-reckoning baseline."""

import unittest
import numpy as np
import pandas as pd

from dataeval.harness.baseline import (
    DEFAULT_PRE_OUTAGE_WINDOW_S,
    PreOutageState,
    extract_pre_outage_state,
    propagate_constant_velocity_heading,
)
from dataeval.harness.metrics import haversine_distance
from dataeval.harness.outage import OutageWindow


class TestBaseline(unittest.TestCase):
    """Test suite for baseline state extraction and constant-velocity propagation."""

    def setUp(self):
        # 10 Hz trajectory moving East along latitude 52.0 at 20 m/s for 30 seconds
        n = 300
        t = np.linspace(0.0, 29.9, n)
        # 1 deg lon at lat 52 is approx 68,679 m
        lon_deg = -1.5 + (20.0 * t) / (6371000.0 * np.cos(np.deg2rad(52.0)) * (np.pi / 180.0))

        self.df = pd.DataFrame({
            "timestamp_s": t,
            "phone_latitude_deg": np.full(n, 52.0),
            "phone_longitude_deg": lon_deg,
            "phone_gps_speed_ms": np.full(n, 20.0, dtype=np.float32),
            "phone_gps_bearing_deg": np.full(n, 90.0, dtype=np.float32),
            "can_latitude_deg": np.full(n, 52.0),
            "can_longitude_deg": lon_deg,
            "can_gps_velocity_kmh": np.full(n, 72.0, dtype=np.float32),
            "can_gps_heading_deg": np.full(n, 90.0, dtype=np.float32),
        })

    def test_extract_pre_outage_state_no_leakage(self):
        """extract_pre_outage_state uses strictly data before window.start_s."""
        window = OutageWindow(
            run_id="test_run",
            source_side="phone",
            start_s=10.0,
            duration_s=10.0,
            end_s=20.0,
        )
        # Corrupt data inside the outage window to test zero-leakage guarantee
        df_corrupted = self.df.copy()
        mask_outage = (df_corrupted["timestamp_s"] >= 10.0) & (df_corrupted["timestamp_s"] < 20.0)
        df_corrupted.loc[mask_outage, "phone_latitude_deg"] = 0.0
        df_corrupted.loc[mask_outage, "phone_gps_speed_ms"] = 999.0

        state = extract_pre_outage_state(df_corrupted, window, pre_outage_window_s=1.0)
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state.lat0, 52.0, places=5)
        self.assertAlmostEqual(state.v0, 20.0, places=2)
        self.assertAlmostEqual(state.theta0, 90.0, places=2)
        self.assertLess(state.t0, 10.0)

    def test_circular_heading_averaging_near_zero(self):
        """Heading averaging across 359° and 1° correctly resolves to 0° (North), not 180°."""
        n = 20
        t = np.linspace(0.0, 1.9, n)
        # Headings alternating around true North (358°, 359°, 0°, 1°, 2°)
        headings = np.array([358.0, 359.0, 0.0, 1.0, 2.0] * 4, dtype=np.float32)

        test_df = pd.DataFrame({
            "timestamp_s": t,
            "phone_latitude_deg": np.full(n, 52.0),
            "phone_longitude_deg": np.full(n, -1.5),
            "phone_gps_speed_ms": np.full(n, 10.0, dtype=np.float32),
            "phone_gps_bearing_deg": headings,
        })
        window = OutageWindow(run_id="wrap_test", source_side="phone", start_s=2.0, duration_s=5.0)
        state = extract_pre_outage_state(test_df, window, pre_outage_window_s=1.0)
        self.assertIsNotNone(state)
        # Circular mean of [358, 359, 0, 1, 2] should be ~0.0° (or 360.0°), not 180°!
        self.assertTrue(state.theta0 < 2.0 or state.theta0 > 358.0)

    def test_straight_line_propagation_zero_error(self):
        """Constant-velocity straight line propagation produces ~0 error against straight ground truth."""
        window = OutageWindow(run_id="straight", source_side="phone", start_s=10.0, duration_s=10.0, end_s=20.0)
        state = extract_pre_outage_state(self.df, window, pre_outage_window_s=1.0)

        target_ts = self.df.loc[(self.df["timestamp_s"] >= 10.0) & (self.df["timestamp_s"] < 20.0), "timestamp_s"].values
        pred_df = propagate_constant_velocity_heading(state, target_ts)

        gt_lat = self.df.loc[(self.df["timestamp_s"] >= 10.0) & (self.df["timestamp_s"] < 20.0), "can_latitude_deg"].values
        gt_lon = self.df.loc[(self.df["timestamp_s"] >= 10.0) & (self.df["timestamp_s"] < 20.0), "can_longitude_deg"].values

        errors = haversine_distance(pred_df["latitude_deg"].values, pred_df["longitude_deg"].values, gt_lat, gt_lon)
        # Maximum error along a straight line over 10s should be negligible (< 0.5m numerical tolerance)
        self.assertLess(np.max(errors), 0.5)

    def test_curved_trajectory_produces_realistic_error(self):
        """A turning vehicle results in significant, realistic drift under constant-heading baseline."""
        n = 200
        t = np.linspace(0.0, 19.9, n)
        # Vehicle turns in a circle at 20 m/s with turn rate omega = 0.1 rad/s (radius = 200m)
        R_circle = 200.0
        omega = 20.0 / R_circle  # 0.1 rad/s
        theta_t = np.deg2rad(90.0) + omega * t  # Heading starts East (90 deg) and turns clockwise

        # Compute positions on local tangent plane
        x = R_circle * np.sin(omega * t)
        y = R_circle * (1.0 - np.cos(omega * t))
        lat = 52.0 + y / (6371000.0 * (np.pi / 180.0))
        lon = -1.5 + x / (6371000.0 * np.cos(np.deg2rad(52.0)) * (np.pi / 180.0))

        curved_df = pd.DataFrame({
            "timestamp_s": t,
            "phone_latitude_deg": lat,
            "phone_longitude_deg": lon,
            "phone_gps_speed_ms": np.full(n, 20.0, dtype=np.float32),
            "phone_gps_bearing_deg": np.rad2deg(theta_t) % 360.0,
            "can_latitude_deg": lat,
            "can_longitude_deg": lon,
            "can_gps_velocity_kmh": np.full(n, 72.0, dtype=np.float32),
            "can_gps_heading_deg": np.rad2deg(theta_t) % 360.0,
        })

        window = OutageWindow(run_id="curved", source_side="phone", start_s=5.0, duration_s=10.0, end_s=15.0)
        state = extract_pre_outage_state(curved_df, window, pre_outage_window_s=1.0)
        target_ts = curved_df.loc[(curved_df["timestamp_s"] >= 5.0) & (curved_df["timestamp_s"] < 15.0), "timestamp_s"].values
        pred_df = propagate_constant_velocity_heading(state, target_ts)

        gt_lat = curved_df.loc[(curved_df["timestamp_s"] >= 5.0) & (curved_df["timestamp_s"] < 15.0), "can_latitude_deg"].values
        gt_lon = curved_df.loc[(curved_df["timestamp_s"] >= 5.0) & (curved_df["timestamp_s"] < 15.0), "can_longitude_deg"].values
        errors = haversine_distance(pred_df["latitude_deg"].values, pred_df["longitude_deg"].values, gt_lat, gt_lon)

        final_err = errors[-1]
        # In a 10s turn at 20 m/s with 0.1 rad/s turn rate:
        # Distance along curve = 200m. Chord = 2 * 200 * sin(0.5) = 191.7m.
        # Constant heading baseline went straight 200m.
        # Drift error should be ~80-100m.
        self.assertGreater(final_err, 50.0)
        self.assertLess(final_err, 350.0)


if __name__ == "__main__":
    unittest.main()
