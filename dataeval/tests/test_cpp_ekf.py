"""Unit tests for Step 19: C++ 15-State ES-EKF port, batch replay, and leaderboard evaluation."""

from pathlib import Path
import unittest
import pandas as pd

from dataeval.harness.run_ekf_replay import (
    CONFIG_EKF,
    CONFIG_CORRECTED,
    CONFIG_BARE,
    CONFIG_CV,
    generate_three_way_report,
)


class TestCppEkf(unittest.TestCase):
    """Test suite validating C++ EKF replay outputs, leaderboard integrity, and 3-way evaluation."""

    def test_ekf_predictions_output_directory(self):
        """Verify C++ EKF predictions directory exists and contains 253 instances."""
        pred_dir = Path("data/processed/cpp_predictions_ekf")
        if not pred_dir.exists():
            self.skipTest("data/processed/cpp_predictions_ekf not present (gitignored in CI)")
        csv_files = list(pred_dir.glob("*.csv"))
        self.assertEqual(len(csv_files), 253, "Must contain exactly 253 EKF prediction CSVs")

    def test_leaderboard_augmented_with_ekf(self):
        """Verify results/leaderboard.csv contains exactly 253 ekf_zupt_nhc_v1 rows and 1,260 total rows."""
        lb_path = Path("results/leaderboard.csv")
        if not lb_path.exists():
            self.skipTest("results/leaderboard.csv does not exist")

        df = pd.read_csv(lb_path)
        ekf_df = df[df["config"] == CONFIG_EKF]
        self.assertEqual(len(ekf_df), 253, f"Must contain exactly 253 {CONFIG_EKF} rows")
        self.assertEqual(len(df), 1260, "Leaderboard must contain exactly 1,260 rows (377 + 377 + 253 + 253)")

        # Validate non-null entries
        self.assertTrue(ekf_df["final_pos_error_m"].notna().all())
        self.assertTrue(ekf_df["outage_s"].isin([10, 30, 60, 120, 180]).all())
        self.assertTrue(ekf_df["run_id"].notna().all())

    def test_three_way_report_generation(self):
        """Verify three-way report generates required sections and tables."""
        lb_path = Path("results/leaderboard.csv")
        if not lb_path.exists():
            self.skipTest("results/leaderboard.csv does not exist")

        report = generate_three_way_report()
        self.assertIn("## 1. Three-Way Median & p95 Performance Comparison", report)
        self.assertIn("## 2. Per-Instance Improved vs. Regressed Breakdown", report)
        self.assertIn("## 3. Investigation of Regressed Instances & `pair_S3c` Analysis", report)

    def test_leaderboard_summary_has_ekf_rows(self):
        """Verify results/leaderboard_summary.csv contains 5 rows for ekf_zupt_nhc_v1."""
        sum_path = Path("results/leaderboard_summary.csv")
        if not sum_path.exists():
            self.skipTest("results/leaderboard_summary.csv does not exist")

        df = pd.read_csv(sum_path)
        ekf_sum = df[df["config"] == CONFIG_EKF]
        self.assertEqual(len(ekf_sum), 5, f"Must contain 5 duration rows for {CONFIG_EKF}")
        self.assertListEqual(sorted(ekf_sum["outage_s"].tolist()), [10, 30, 60, 120, 180])


if __name__ == "__main__":
    unittest.main()
