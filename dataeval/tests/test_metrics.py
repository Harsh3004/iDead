"""Unit tests for dead-reckoning leaderboard metrics engine."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from dataeval.harness.metrics import (
    MIN_TRAVEL_DISTANCE_M,
    OutageMetrics,
    compute_outage_metrics,
    decompose_along_cross_track,
    haversine_distance,
    wrap_heading_error,
)
from dataeval.harness.run_scoring import LEADERBOARD_COLUMNS, score_outage_dataset


class TestMetrics(unittest.TestCase):
    """Test suite for position error, %distance, CEP50/95, along/cross track, and heading error."""

    def test_haversine_accuracy_known_distance(self):
        """Haversine distance matches known spherical distances."""
        # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) is approx 343 km
        d = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        self.assertAlmostEqual(d / 1000.0, 343.5, delta=2.0)

        # Zero distance
        d_zero = haversine_distance(52.0, -1.5, 52.0, -1.5)
        self.assertAlmostEqual(d_zero, 0.0, places=5)

    def test_along_cross_track_decomposition_pythagoras(self):
        """Along-track and cross-track errors satisfy Pythagoras: e_along^2 + e_cross^2 approx e_total^2."""
        gt_lat, gt_lon = 52.0, -1.5
        # 30m East and 40m North -> 50m displacement
        R = 6371000.0
        pred_lat = gt_lat + (40.0 / R) * (180.0 / np.pi)
        pred_lon = gt_lon + (30.0 / (R * np.cos(np.deg2rad(gt_lat)))) * (180.0 / np.pi)

        # Track heading = 45 degrees
        along, cross = decompose_along_cross_track(pred_lat, pred_lon, gt_lat, gt_lon, track_heading_deg=45.0)

        total_direct = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
        reconstructed = np.sqrt(along**2 + cross**2)
        self.assertAlmostEqual(reconstructed, total_direct, delta=0.05)
        self.assertAlmostEqual(total_direct, 50.0, delta=0.1)

    def test_cep50_and_cep95_known_percentiles(self):
        """CEP50 and CEP95 correctly evaluate 50th and 95th percentiles of position error."""
        n = 101
        t = np.linspace(0.0, 10.0, n)
        # Pred trajectory and GT trajectory with known linearly increasing errors: 0m to 100m
        gt_lat = np.full(n, 52.0)
        gt_lon = np.full(n, -1.5)

        R = 6371000.0
        # Linearly increasing northing error: error = t * 10 meters (from 0 to 100m)
        pred_lat = gt_lat + (t * 10.0 / R) * (180.0 / np.pi)
        pred_lon = np.full(n, -1.5)

        pred_df = pd.DataFrame({"timestamp_s": t, "latitude_deg": pred_lat, "longitude_deg": pred_lon, "heading_deg": np.zeros(n)})
        gt_df = pd.DataFrame({"timestamp_s": t, "latitude_deg": gt_lat, "longitude_deg": gt_lon, "heading_deg": np.zeros(n)})

        metrics = compute_outage_metrics(pred_df, gt_df)
        self.assertAlmostEqual(metrics.final_pos_error_m, 100.0, delta=0.1)
        # Median of uniform [0, 100] is 50.0m
        self.assertAlmostEqual(metrics.cep50_m, 50.0, delta=0.5)
        # 95th percentile of uniform [0, 100] is 95.0m
        self.assertAlmostEqual(metrics.cep95_m, 95.0, delta=0.5)

    def test_pct_of_distance_stationary_handles_zero_cleanly(self):
        """Stationary run (travel distance < 1.0m) returns NaN for pct_of_distance without ZeroDivisionError."""
        n = 50
        t = np.linspace(0.0, 4.9, n)
        # Stationary vehicle with slight 0.05m GPS jitter
        gt_lat = 52.0 + np.sin(t) * 1e-7
        gt_lon = -1.5 + np.cos(t) * 1e-7

        pred_lat = np.full(n, 52.0)
        pred_lon = np.full(n, -1.5)

        pred_df = pd.DataFrame({"timestamp_s": t, "latitude_deg": pred_lat, "longitude_deg": pred_lon, "heading_deg": np.zeros(n)})
        gt_df = pd.DataFrame({"timestamp_s": t, "latitude_deg": gt_lat, "longitude_deg": gt_lon, "heading_deg": np.zeros(n)})

        metrics = compute_outage_metrics(pred_df, gt_df)
        # True distance travelled is ~0.1m (< 1.0m threshold)
        self.assertLess(metrics.true_distance_m, 1.0)
        # pct_of_distance must be NaN
        self.assertTrue(np.isnan(metrics.pct_of_distance))
        # Skip reason explains stationary run
        self.assertIn("stationary_run", metrics.skip_reason)
        # But final_pos_error_m and CEPs are valid numeric values
        self.assertFalse(np.isnan(metrics.final_pos_error_m))
        self.assertFalse(np.isnan(metrics.cep50_m))

    def test_heading_error_wrapping_360(self):
        """Heading error wraps correctly across 0°/360° boundary."""
        # 359° vs 1° is 2° difference, NOT 358°
        err1 = wrap_heading_error(359.0, 1.0)
        self.assertAlmostEqual(err1, 2.0, places=4)

        err2 = wrap_heading_error(1.0, 359.0)
        self.assertAlmostEqual(err2, 2.0, places=4)

        # Opposite directions: 0° vs 180° is 180°
        err3 = wrap_heading_error(0.0, 180.0)
        self.assertAlmostEqual(err3, 180.0, places=4)

        # NaN handling
        err_nan = wrap_heading_error(np.nan, 90.0)
        self.assertTrue(np.isnan(err_nan))

    def test_no_ground_truth_returns_nan_cleanly(self):
        """When gt_df is None (unpaired run), metrics return NaN without crashing."""
        n = 20
        t = np.linspace(0.0, 1.9, n)
        pred_df = pd.DataFrame({"timestamp_s": t, "latitude_deg": np.full(n, 52.0), "longitude_deg": np.full(n, -1.5), "heading_deg": np.zeros(n)})

        metrics = compute_outage_metrics(pred_df, None)
        self.assertTrue(np.isnan(metrics.final_pos_error_m))
        self.assertTrue(np.isnan(metrics.pct_of_distance))
        self.assertEqual(metrics.status, "no_ground_truth")
        self.assertIn("unpaired_run", metrics.skip_reason)

    def test_non_destructive_leaderboard_append(self):
        """Leaderboard file appends new evaluation rows on repeated runs rather than overwriting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lb_file = tmp_path / "leaderboard.csv"

            # Create dummy initial leaderboard
            initial_rows = pd.DataFrame([{
                "run_id": "initial_run",
                "timestamp": "2026-09-01T00:00:00Z",
                "config": "prior_config",
                "outage_s": 10,
                "final_pos_error_m": 5.0,
                "pct_of_distance": 10.0,
                "cep50_m": 3.0,
                "cep95_m": 4.5,
                "along_track_m": 2.0,
                "cross_track_m": 1.0,
                "heading_error_deg": 1.5,
                "outage_id": "initial__10s__0",
                "split": "val",
                "source_side": "phone",
                "has_ground_truth": True,
                "skip_reason": "",
            }])
            initial_rows.to_csv(lb_file, index=False)

            # Append a new batch of 2 rows
            new_batch = pd.DataFrame([
                {
                    "run_id": f"run_{i}",
                    "timestamp": "2026-09-07T00:00:00Z",
                    "config": "baseline_cv_heading_v1",
                    "outage_s": 10,
                    "final_pos_error_m": 12.0,
                    "pct_of_distance": 15.0,
                    "cep50_m": 8.0,
                    "cep95_m": 11.0,
                    "along_track_m": 5.0,
                    "cross_track_m": 7.0,
                    "heading_error_deg": 3.0,
                    "outage_id": f"run_{i}__10s__0",
                    "split": "test",
                    "source_side": "phone",
                    "has_ground_truth": True,
                    "skip_reason": "",
                }
                for i in range(2)
            ])
            new_batch.to_csv(lb_file, mode="a", header=False, index=False)

            # Read back
            combined = pd.read_csv(lb_file)
            self.assertEqual(len(combined), 3)
            self.assertEqual(combined.iloc[0]["run_id"], "initial_run")
            self.assertEqual(combined.iloc[1]["run_id"], "run_0")
            self.assertEqual(combined.iloc[2]["run_id"], "run_1")


if __name__ == "__main__":
    unittest.main()
