"""Unit tests for Step 19, 20 & 22: C++ 15-State ES-EKF port, batch replay, tuning, and leaderboard evaluation."""

from pathlib import Path
import unittest
import pandas as pd

from dataeval.harness.run_ekf_replay import (
    CONFIG_EKF,
    CONFIG_EKF_V2,
    CONFIG_EKF_V3,
    CONFIG_CORRECTED,
    CONFIG_BARE,
    CONFIG_CV,
    generate_three_way_report,
    generate_step20_report,
    generate_step22_report,
)


class TestCppEkf(unittest.TestCase):
    """Test suite validating C++ EKF replay outputs, leaderboard integrity, and comparative evaluation."""

    def test_ekf_predictions_output_directory(self):
        """Verify C++ EKF predictions directory exists and contains 253 instances."""
        pred_dir = Path("data/processed/cpp_predictions_ekf")
        if not pred_dir.exists():
            self.skipTest("data/processed/cpp_predictions_ekf not present (gitignored in CI)")
        csv_files = list(pred_dir.glob("*.csv"))
        self.assertEqual(len(csv_files), 253, "Must contain exactly 253 EKF prediction CSVs")

    def test_leaderboard_augmented_with_ekf(self):
        """Verify results/leaderboard.csv contains exactly 253 ekf_zupt_nhc_v1 rows and 1,260, 1,513, or 1,766 total rows."""
        lb_path = Path("results/leaderboard.csv")
        if not lb_path.exists():
            self.skipTest("results/leaderboard.csv does not exist")

        df = pd.read_csv(lb_path)
        ekf_df = df[df["config"] == CONFIG_EKF]
        self.assertEqual(len(ekf_df), 253, f"Must contain exactly 253 {CONFIG_EKF} rows")
        self.assertIn(len(df), [1260, 1513, 1766], "Leaderboard must contain 1,260 (v1), 1,513 (v2), or 1,766 (v3) rows")

        # Validate non-null entries
        self.assertTrue(ekf_df["final_pos_error_m"].notna().all())
        self.assertTrue(ekf_df["outage_s"].isin([10, 30, 60, 120, 180]).all())
        self.assertTrue(ekf_df["run_id"].notna().all())

        # If v2 rows present, validate v2
        if (df["config"] == CONFIG_EKF_V2).any():
            ekf_v2_df = df[df["config"] == CONFIG_EKF_V2]
            self.assertEqual(len(ekf_v2_df), 253, f"Must contain exactly 253 {CONFIG_EKF_V2} rows")
            self.assertTrue(ekf_v2_df["final_pos_error_m"].notna().all())
            self.assertTrue(ekf_v2_df["outage_s"].isin([10, 30, 60, 120, 180]).all())
            self.assertTrue(ekf_v2_df["run_id"].notna().all())

        # If v3 rows present, validate v3
        if (df["config"] == CONFIG_EKF_V3).any():
            ekf_v3_df = df[df["config"] == CONFIG_EKF_V3]
            self.assertEqual(len(ekf_v3_df), 253, f"Must contain exactly 253 {CONFIG_EKF_V3} rows")
            self.assertTrue(ekf_v3_df["final_pos_error_m"].notna().all())
            self.assertTrue(ekf_v3_df["outage_s"].isin([10, 30, 60, 120, 180]).all())
            self.assertTrue(ekf_v3_df["run_id"].notna().all())

    def test_three_way_report_generation(self):
        """Verify three-way report generates required sections and tables."""
        lb_path = Path("results/leaderboard.csv")
        if not lb_path.exists():
            self.skipTest("results/leaderboard.csv does not exist")

        report = generate_three_way_report(report_path=None)
        self.assertIn("## 1. Three-Way Median & p95 Performance Comparison", report)
        self.assertIn("## 2. Per-Instance Improved vs. Regressed Breakdown", report)
        self.assertIn("## 3. Investigation of Regressed Instances & `pair_S3c` Analysis", report)

    def test_step20_report_generation(self):
        """Verify Step 20 tuning report generates required sections and tables."""
        lb_path = Path("results/leaderboard.csv")
        if not lb_path.exists():
            self.skipTest("results/leaderboard.csv does not exist")

        df = pd.read_csv(lb_path)
        if (df["config"] == CONFIG_EKF_V2).any():
            report = generate_step20_report(report_path=None)
            self.assertIn("## 1. Executive Summary & Root-Cause Diagnosis", report)
            self.assertIn("## 2. Five-Way Median & p95 Performance Comparison", report)
            self.assertIn("## 3. 10s Short-Horizon Outage Deep Dive", report)
            self.assertIn("## 4. `pair_S3c` High-Speed Curvature Deep Dive", report)
            self.assertIn("## 5. Side-Effect Audit Across 30s", report)

    def test_step22_report_generation(self):
        """Verify Step 22 kinematic report generates required sections and tables."""
        lb_path = Path("results/leaderboard.csv")
        if not lb_path.exists():
            self.skipTest("results/leaderboard.csv does not exist")

        df = pd.read_csv(lb_path)
        if (df["config"] == CONFIG_EKF_V3).any():
            report = generate_step22_report(report_path=None)
            self.assertIn("## 1. Executive Summary & Physics-First Root Cause Resolution", report)
            self.assertIn("## 2. Six-Way Median & p95 Performance Comparison", report)
            self.assertIn("## 3. Fleetwide Head-to-Head Win-Rate Comparisons", report)
            self.assertIn("## 4. 10s Short-Horizon Cohort Confirmation", report)
            self.assertIn("## 5. Genuine Turn Relaxation Preserved", report)
            self.assertIn("## 6. Straight-Road Vibration Immunity on Previously Regressed Instances", report)
            self.assertIn("## 7. Dead-Reckoned Speed Dependency & Stability Risk Audit", report)

    def test_leaderboard_summary_has_ekf_rows(self):
        """Verify results/leaderboard_summary.csv contains 5 rows for each EKF config present."""
        sum_path = Path("results/leaderboard_summary.csv")
        if not sum_path.exists():
            self.skipTest("results/leaderboard_summary.csv does not exist")

        df = pd.read_csv(sum_path)
        ekf_sum = df[df["config"] == CONFIG_EKF]
        self.assertEqual(len(ekf_sum), 5, f"Must contain 5 duration rows for {CONFIG_EKF}")
        self.assertListEqual(sorted(ekf_sum["outage_s"].tolist()), [10, 30, 60, 120, 180])

        if (df["config"] == CONFIG_EKF_V2).any():
            v2_sum = df[df["config"] == CONFIG_EKF_V2]
            self.assertEqual(len(v2_sum), 5, f"Must contain 5 duration rows for {CONFIG_EKF_V2}")
            self.assertListEqual(sorted(v2_sum["outage_s"].tolist()), [10, 30, 60, 120, 180])

        if (df["config"] == CONFIG_EKF_V3).any():
            v3_sum = df[df["config"] == CONFIG_EKF_V3]
            self.assertEqual(len(v3_sum), 5, f"Must contain 5 duration rows for {CONFIG_EKF_V3}")
            self.assertListEqual(sorted(v3_sum["outage_s"].tolist()), [10, 30, 60, 120, 180])


if __name__ == "__main__":
    unittest.main()

