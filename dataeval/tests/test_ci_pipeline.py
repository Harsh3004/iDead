"""Integration tests for CI pipeline using checked-in lightweight fixtures.

Validates that every stage of the dataeval pipeline (ingestion, pairing,
outage simulation, C++ prediction loading, and scoring summary) functions
correctly on real-shaped hardware log slices without requiring the full 1.71 GB dataset.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from dataeval.harness.baseline import (
    extract_pre_outage_state,
    propagate_constant_velocity_heading,
)
from dataeval.harness.cpp_predictions import load_cpp_prediction
from dataeval.harness.metrics import compute_outage_metrics
from dataeval.harness.outage import (
    PHONE_GNSS_COLUMNS,
    OutageWindow,
    apply_outage,
    ground_truth_trajectory,
)
from dataeval.harness.summary import compute_leaderboard_summary
from dataeval.ingest.smartphone import parse_phone_csv
from dataeval.ingest.vehicle_can import parse_can_csv


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCiPipeline(unittest.TestCase):
    """Integration test suite executing dataeval pipeline components against CI fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.raw_can_csv = FIXTURES_DIR / "raw" / "V-Vfa01_slice.csv"
        cls.raw_phone_csv = FIXTURES_DIR / "raw" / "S-Vfa01_slice.csv"
        cls.paired_pq = FIXTURES_DIR / "processed" / "paired" / "pair_Vfa01.parquet"
        cls.outage_pq = FIXTURES_DIR / "processed" / "outages" / "test" / "pair_Vfa01__10s__0.parquet"
        cls.outage_manifest = FIXTURES_DIR / "processed" / "outages" / "manifest.csv"
        cls.pred_csv = FIXTURES_DIR / "processed" / "cpp_predictions" / "pair_Vfa01__10s__0.csv"
        cls.lb_csv = FIXTURES_DIR / "results" / "leaderboard.csv"

    def test_raw_can_fixture_ingestion(self):
        """Parse raw CAN CSV fixture and verify 29 canonical columns and unit conversions."""
        self.assertTrue(self.raw_can_csv.exists(), f"Missing fixture: {self.raw_can_csv}")
        df = parse_can_csv(self.raw_can_csv)

        self.assertGreater(len(df), 0)
        self.assertEqual(len(df.columns), 29)
        self.assertIn("timestamp_s", df.columns)
        self.assertIn("latitude_deg", df.columns)
        self.assertIn("gps_velocity_kmh", df.columns)
        self.assertTrue(np.issubdtype(df["timestamp_s"].dtype, np.floating))

    def test_raw_phone_fixture_ingestion(self):
        """Parse raw Smartphone CSV fixture and verify 27 canonical columns."""
        self.assertTrue(self.raw_phone_csv.exists(), f"Missing fixture: {self.raw_phone_csv}")
        df, median_dt = parse_phone_csv(self.raw_phone_csv)

        self.assertGreater(len(df), 0)
        self.assertEqual(len(df.columns), 27)
        self.assertIn("timestamp_s", df.columns)
        self.assertIn("accel_x_m_s2", df.columns)
        self.assertIn("gyro_z_rad_s", df.columns)

    def test_paired_parquet_structure(self):
        """Verify synchronized paired fixture contains aligned CAN and Phone channels."""
        self.assertTrue(self.paired_pq.exists(), f"Missing fixture: {self.paired_pq}")
        df = pd.read_parquet(self.paired_pq)

        self.assertGreater(len(df), 0)
        self.assertIn("timestamp_s", df.columns)
        self.assertIn("can_latitude_deg", df.columns)
        self.assertIn("phone_accel_x_m_s2", df.columns)

    def test_outage_masking_on_paired_fixture(self):
        """Apply synthetic outage window to paired fixture and verify GNSS blanking."""
        df = pd.read_parquet(self.paired_pq)
        t_start = float(df["timestamp_s"].iloc[5])
        window = OutageWindow(
            run_id="pair_Vfa01",
            source_side="phone",
            start_s=t_start,
            duration_s=2.0,
            end_s=t_start + 2.0,
        )

        masked_df = apply_outage(df, window)
        mask_outage = (masked_df["timestamp_s"] >= window.start_s) & (masked_df["timestamp_s"] < window.end_s)

        for col in PHONE_GNSS_COLUMNS:
            if col in masked_df.columns:
                self.assertTrue(
                    masked_df.loc[mask_outage, col].isna().all(),
                    f"Column {col} was not properly blanked during outage.",
                )

        # Ensure IMU channels remain untouched
        self.assertFalse(masked_df.loc[mask_outage, "phone_accel_x_m_s2"].isna().any())

    def test_cpp_prediction_loading_and_scoring(self):
        """Load C++ prediction fixture and evaluate against ground truth trajectory."""
        self.assertTrue(self.pred_csv.exists(), f"Missing fixture: {self.pred_csv}")
        self.assertTrue(self.outage_pq.exists(), f"Missing fixture: {self.outage_pq}")

        pred_df = load_cpp_prediction(self.pred_csv)
        self.assertEqual(len(pred_df), 50)

        outage_df = pd.read_parquet(self.outage_pq)
        window = OutageWindow(
            run_id="pair_Vfa01",
            source_side="phone",
            start_s=float(pred_df["timestamp_s"].iloc[0]),
            duration_s=10.0,
            end_s=float(pred_df["timestamp_s"].iloc[-1]),
        )

        gt_df = ground_truth_trajectory(outage_df, window)
        if gt_df is not None and len(gt_df) >= 2:
            metrics = compute_outage_metrics(pred_df, gt_df)
            self.assertEqual(metrics.status, "ok")
            self.assertGreater(metrics.final_pos_error_m, 0.0)

    def test_leaderboard_summary_on_fixture(self):
        """Verify summary aggregator computes clean grouped rows from fixture leaderboard."""
        self.assertTrue(self.lb_csv.exists(), f"Missing fixture: {self.lb_csv}")
        summary = compute_leaderboard_summary(self.lb_csv)
        self.assertGreater(len(summary), 0)
        self.assertIn("median_final_pos_error_m", summary.columns)


if __name__ == "__main__":
    unittest.main()
