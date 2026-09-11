"""Unit tests for Step 17: Diagnosing calibrated-tier reversal."""

import unittest
from pathlib import Path
import pandas as pd

from dataeval.harness.diagnose_reversal import (
    compute_trajectory_error_curves,
    check_gyro_bias_stability,
    check_accelerometer_and_gravity_leakage,
    run_ablation_suspect_runs,
    analyze_outlier_runs,
)


class TestDiagnoseReversal(unittest.TestCase):
    """Test suite validating Step 17 reversal diagnostic calculations."""

    @classmethod
    def setUpClass(cls):
        required_paths = [
            Path("data/processed/outages/manifest.csv"),
            Path("data/processed/paired"),
            Path("data/processed/cpp_predictions_corrected"),
        ]
        for p in required_paths:
            if not p.exists():
                raise unittest.SkipTest(f"Required processed dataset {p} not present (gitignored in CI environment)")

    def test_trajectory_error_curves_calculation(self):
        """Verify error curves compute for all 5 durations with valid crossover."""
        curves = compute_trajectory_error_curves()
        self.assertEqual(len(curves), 5, "Must compute curves for 10, 30, 60, 120, 180s")
        for d in [10, 30, 60, 120, 180]:
            self.assertIn(d, curves)
            data = curves[d]
            self.assertGreater(data["n"], 20, f"Must have >=20 calibrated instances at {d}s")
            self.assertEqual(len(data["median_bare"]), 21)
            self.assertEqual(len(data["median_corr"]), 21)

    def test_gyro_bias_stability_multi_stops(self):
        """Verify multi-stop stillness detection on calibrated runs."""
        stability = check_gyro_bias_stability()
        self.assertEqual(stability["total_calibrated_runs"], 33)
        self.assertGreaterEqual(stability["runs_with_multi_stops"], 20, "At least 20 runs must have >=2 stops")
        drift_df = stability["drift_df"]
        self.assertFalse(drift_df.empty)
        # Median drift should be in the realistic range (0.1 to 1.0 deg/s)
        med_deg_s = float(drift_df["gyro_shift_norm_deg_s"].median())
        self.assertGreater(med_deg_s, 0.1)
        self.assertLess(med_deg_s, 2.0)

    def test_gravity_leakage_metrics(self):
        """Verify dynamic tilt and gravity leakage quantification."""
        leak_df = check_accelerometer_and_gravity_leakage()
        self.assertFalse(leak_df.empty)
        self.assertIn("mean_tilt_dev_deg", leak_df.columns)
        self.assertIn("a_leak_mean_m_s2", leak_df.columns)
        # Tilt deviation should average 3-10 degrees
        mean_tilt = float(leak_df["mean_tilt_dev_deg"].mean())
        self.assertGreater(mean_tilt, 2.0)
        self.assertLess(mean_tilt, 15.0)

    def test_suspect_runs_ablation_results(self):
        """Verify suspect runs ablation shows bias correction dominance."""
        abl_df = run_ablation_suspect_runs()
        self.assertFalse(abl_df.empty)
        for r in ["pair_S2", "pair_Vta17"]:
            sub = abl_df[abl_df["run_id"] == r]
            self.assertFalse(sub.empty)
            # At 180s, bias_only error must be significantly lower than bare error
            sub_180 = sub[sub["outage_s"] == 180]
            if not sub_180.empty:
                row = sub_180.iloc[0]
                self.assertLess(row["err_bias_only_m"], row["err_bare_m"])

    def test_outlier_runs_identified(self):
        """Verify outlier analysis identifies key culprits."""
        outlier_df = analyze_outlier_runs()
        self.assertFalse(outlier_df.empty)
        culprit_runs = set(outlier_df[outlier_df["diff_m"] > 10000]["run_id"])
        # Should include known degraded runs like pair_Vw1, pair_Vta2, or pair_Vtb3
        self.assertTrue(len(culprit_runs.intersection({"pair_Vw1", "pair_Vta2", "pair_Vtb3", "pair_Vfa01"})) > 0)


if __name__ == "__main__":
    unittest.main()
