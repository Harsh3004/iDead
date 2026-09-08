"""Unit tests for leaderboard summary aggregation logic."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from dataeval.harness.summary import (
    SUMMARY_COLUMNS,
    compute_leaderboard_summary,
)


class TestLeaderboardSummary(unittest.TestCase):
    """Test suite for leaderboard summary aggregation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a small controlled synthetic leaderboard
        # Config A, 10s: 5 paired rows [10, 20, 30, 40, 50] + 1 unpaired row [9999]
        # Config B, 30s: 3 paired rows [100, 200, 300]
        data = [
            # Config A (10s)
            {"config": "cfg_a", "outage_s": 10, "final_pos_error_m": 10.0, "pct_of_distance": 10.0, "heading_error_deg": 1.0, "has_ground_truth": True},
            {"config": "cfg_a", "outage_s": 10, "final_pos_error_m": 20.0, "pct_of_distance": 20.0, "heading_error_deg": 2.0, "has_ground_truth": True},
            {"config": "cfg_a", "outage_s": 10, "final_pos_error_m": 30.0, "pct_of_distance": 30.0, "heading_error_deg": 3.0, "has_ground_truth": True},
            {"config": "cfg_a", "outage_s": 10, "final_pos_error_m": 40.0, "pct_of_distance": 40.0, "heading_error_deg": 4.0, "has_ground_truth": True},
            {"config": "cfg_a", "outage_s": 10, "final_pos_error_m": 50.0, "pct_of_distance": 50.0, "heading_error_deg": 5.0, "has_ground_truth": True},
            # Unpaired row that MUST be filtered out
            {"config": "cfg_a", "outage_s": 10, "final_pos_error_m": 9999.0, "pct_of_distance": 999.0, "heading_error_deg": 99.0, "has_ground_truth": False},
            # Config B (30s) with one NaN in pct_of_distance (stationary run)
            {"config": "cfg_b", "outage_s": 30, "final_pos_error_m": 100.0, "pct_of_distance": 100.0, "heading_error_deg": 10.0, "has_ground_truth": True},
            {"config": "cfg_b", "outage_s": 30, "final_pos_error_m": 200.0, "pct_of_distance": np.nan, "heading_error_deg": 20.0, "has_ground_truth": True},
            {"config": "cfg_b", "outage_s": 30, "final_pos_error_m": 300.0, "pct_of_distance": 300.0, "heading_error_deg": 30.0, "has_ground_truth": True},
        ]
        self.synthetic_csv = self.temp_path / "synthetic_leaderboard.csv"
        pd.DataFrame(data).to_csv(self.synthetic_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_summary_aggregation_synthetic(self):
        """Aggregation correctly filters unpaired rows and computes exact medians/p95."""
        out_csv = self.temp_path / "summary.csv"
        summary = compute_leaderboard_summary(self.synthetic_csv, output_path=out_csv)

        self.assertEqual(len(summary), 2)
        self.assertListEqual(summary.columns.tolist(), SUMMARY_COLUMNS)

        # Check cfg_a (10s): 5 paired rows, median must be 30.0, 9999.0 ignored
        row_a = summary[summary["config"] == "cfg_a"].iloc[0]
        self.assertEqual(row_a["outage_s"], 10)
        self.assertEqual(row_a["n_paired"], 5)
        self.assertAlmostEqual(row_a["median_final_pos_error_m"], 30.0)
        self.assertAlmostEqual(row_a["median_pct_of_distance"], 30.0)
        self.assertAlmostEqual(row_a["median_heading_error_deg"], 3.0)
        self.assertAlmostEqual(row_a["p95_final_pos_error_m"], float(pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]).quantile(0.95)))

        # Check cfg_b (30s): 3 paired rows, pct_of_distance median computed ignoring NaN
        row_b = summary[summary["config"] == "cfg_b"].iloc[0]
        self.assertEqual(row_b["outage_s"], 30)
        self.assertEqual(row_b["n_paired"], 3)
        self.assertAlmostEqual(row_b["median_final_pos_error_m"], 200.0)
        self.assertAlmostEqual(row_b["median_pct_of_distance"], 200.0)  # median([100, 300]) = 200

        # Verify output file was written
        self.assertTrue(out_csv.exists())

    def test_summary_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            compute_leaderboard_summary(self.temp_path / "missing.csv")

    def test_summary_missing_columns(self):
        """Leaderboard missing required columns raises ValueError."""
        bad_csv = self.temp_path / "bad.csv"
        pd.DataFrame({"config": ["a"], "outage_s": [10]}).to_csv(bad_csv, index=False)
        with self.assertRaises(ValueError) as ctx:
            compute_leaderboard_summary(bad_csv)
        self.assertIn("missing required columns", str(ctx.exception))

    def test_summary_against_real_leaderboard(self):
        """Aggregation on real results/leaderboard.csv yields exact known counts and medians."""
        real_lb = Path("results/leaderboard.csv")
        if not real_lb.exists():
            self.skipTest("results/leaderboard.csv does not exist.")

        summary = compute_leaderboard_summary(real_lb)
        # 2 configs x 5 outage durations = 10 rows
        self.assertEqual(len(summary), 10)

        # Paired cohort sizes per duration must match [67, 58, 52, 41, 35]
        expected_counts = {10: 67, 30: 58, 60: 52, 120: 41, 180: 35}
        for cfg in ["baseline_cv_heading_v1", "baseline_cpp_strapdown_v1"]:
            sub = summary[summary["config"] == cfg]
            self.assertEqual(len(sub), 5)
            for _, r in sub.iterrows():
                d = int(r["outage_s"])
                self.assertEqual(r["n_paired"], expected_counts[d])

        # Verify known medians for CV and Strapdown at 10s and 180s
        cv_10 = summary[(summary["config"] == "baseline_cv_heading_v1") & (summary["outage_s"] == 10)].iloc[0]
        self.assertAlmostEqual(cv_10["median_final_pos_error_m"], 34.1126, places=2)

        cpp_10 = summary[(summary["config"] == "baseline_cpp_strapdown_v1") & (summary["outage_s"] == 10)].iloc[0]
        self.assertAlmostEqual(cpp_10["median_final_pos_error_m"], 87.0982, places=2)

        cpp_180 = summary[(summary["config"] == "baseline_cpp_strapdown_v1") & (summary["outage_s"] == 180)].iloc[0]
        self.assertAlmostEqual(cpp_180["median_final_pos_error_m"], 47929.0214, places=1)


if __name__ == "__main__":
    unittest.main()
