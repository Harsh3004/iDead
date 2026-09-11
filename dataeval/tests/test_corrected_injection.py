"""Unit tests for Step 16: Initial attitude and gyro bias injection into C++ strapdown replay."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from dataeval.harness.dynamic_phase import construct_attitude_quaternion
from dataeval.harness.run_corrected_replay import (
    load_module_b_tiers,
    compute_tiered_comparison_report,
    CONFIG_CORRECTED,
    CONFIG_CPP_BASE,
)


class TestCorrectedInjection(unittest.TestCase):
    """Test suite for initial attitude and gyro bias injection."""

    def test_module_b_tiers_coverage(self):
        """Verify that all 72 paired runs are mapped to valid tiers."""
        csv_path = Path("results/module_b_initial_attitude.csv")
        self.assertTrue(csv_path.exists(), "module_b_initial_attitude.csv must exist")

        tiers = load_module_b_tiers(csv_path)
        self.assertEqual(len(tiers), 72, "Must contain exactly 72 paired runs")

        calibrated_count = sum(1 for v in tiers.values() if v["tier"] == "module_a_measured")
        fallback_count = sum(1 for v in tiers.values() if v["tier"] == "fallback_zero")

        self.assertEqual(calibrated_count, 33, "Must have exactly 33 module_a_measured runs")
        self.assertEqual(fallback_count, 39, "Must have exactly 39 fallback_zero runs")

    def test_quaternion_synthesis_properties(self):
        """Verify attitude quaternion synthesis matches ENU convention."""
        # East heading (90 deg), flat leveling
        q_east = construct_attitude_quaternion(yaw_rad=np.pi / 2.0, pitch_rad=0.0, roll_rad=0.0)
        qw, qx, qy, qz = q_east
        self.assertAlmostEqual(qw, np.cos(np.pi / 4.0), places=5)
        self.assertAlmostEqual(qx, 0.0, places=5)
        self.assertAlmostEqual(qy, 0.0, places=5)
        self.assertAlmostEqual(qz, -np.sin(np.pi / 4.0), places=5)

        # Norm must be 1.0
        norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_prediction_output_directory(self):
        """Verify corrected predictions directory exists and contains 253 instances."""
        pred_dir = Path("data/processed/cpp_predictions_corrected")
        if not pred_dir.exists():
            self.skipTest("data/processed/cpp_predictions_corrected does not exist in environment (gitignored in CI)")
        csv_files = list(pred_dir.glob("*.csv"))
        self.assertEqual(len(csv_files), 253, "Must contain exactly 253 corrected prediction CSVs")

    def test_leaderboard_augmented_correctly(self):
        """Verify results/leaderboard.csv contains exactly 253 corrected rows."""
        lb_path = Path("results/leaderboard.csv")
        self.assertTrue(lb_path.exists(), "leaderboard.csv must exist")

        df = pd.read_csv(lb_path)
        corr_df = df[df["config"] == CONFIG_CORRECTED]
        self.assertEqual(len(corr_df), 253, "Must contain exactly 253 corrected_init_strapdown_v1 rows")

        # Every corrected row must have valid outage_s and run_id
        self.assertTrue(corr_df["run_id"].notna().all())
        self.assertTrue(corr_df["outage_s"].isin([10, 30, 60, 120, 180]).all())
        self.assertTrue(corr_df["final_pos_error_m"].notna().all())

    def test_tiered_comparison_report_sections(self):
        """Verify report generation contains all required sub-population sections."""
        report = compute_tiered_comparison_report()
        self.assertIn("## 1. All Eligible Paired Instances (N=253)", report)
        self.assertIn("## 2. Calibrated Tier: `module_a_measured`", report)
        self.assertIn("## 3. Fallback Tier: `fallback_zero`", report)
        self.assertIn("## 4. Suspect Bias Runs Callout: `pair_S2` and `pair_Vta17`", report)


if __name__ == "__main__":
    unittest.main()
