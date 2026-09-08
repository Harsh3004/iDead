"""Unit tests for C++ strapdown INS prediction loader and validator."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from dataeval.harness.cpp_predictions import (
    load_cpp_prediction,
    spot_check_predictions,
    validate_prediction_against_manifest,
)


class TestCppPredictions(unittest.TestCase):
    """Test suite for loading and validating C++ prediction CSV files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Standard valid 10 Hz trajectory over 10s
        n = 100
        t = np.linspace(100.0, 109.9, n)
        self.valid_csv = self.temp_path / "valid_outage.csv"
        df = pd.DataFrame({
            "timestamp_s": t,
            "latitude_deg": np.linspace(52.0, 52.001, n),
            "longitude_deg": np.linspace(-1.5, -1.499, n),
            "speed_ms": np.full(n, 15.0),
            "heading_deg": np.full(n, 45.0),
        })
        df.to_csv(self.valid_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_prediction(self):
        """Valid prediction CSV loads with correct shape and types."""
        df = load_cpp_prediction(self.valid_csv)
        self.assertEqual(len(df), 100)
        self.assertListEqual(
            df.columns.tolist(),
            ["timestamp_s", "latitude_deg", "longitude_deg", "speed_ms", "heading_deg"],
        )
        self.assertAlmostEqual(float(df["timestamp_s"].iloc[0]), 100.0)
        self.assertAlmostEqual(float(df["latitude_deg"].iloc[0]), 52.0)
        self.assertAlmostEqual(float(df["speed_ms"].iloc[0]), 15.0)

    def test_load_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_cpp_prediction(self.temp_path / "non_existent.csv")

    def test_load_empty_file(self):
        """Empty file raises ValueError."""
        empty_file = self.temp_path / "empty.csv"
        empty_file.touch()
        with self.assertRaises(ValueError):
            load_cpp_prediction(empty_file)

    def test_load_missing_columns(self):
        """CSV missing required columns raises ValueError."""
        bad_csv = self.temp_path / "bad_cols.csv"
        df = pd.DataFrame({"timestamp_s": [1.0, 2.0], "latitude_deg": [52.0, 52.1]})
        df.to_csv(bad_csv, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_cpp_prediction(bad_csv)
        self.assertIn("missing required columns", str(ctx.exception))

    def test_load_short_trajectory(self):
        """Trajectory with fewer than 2 points raises ValueError."""
        short_csv = self.temp_path / "short.csv"
        df = pd.DataFrame({
            "timestamp_s": [100.0],
            "latitude_deg": [52.0],
            "longitude_deg": [-1.5],
            "speed_ms": [10.0],
            "heading_deg": [90.0],
        })
        df.to_csv(short_csv, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_cpp_prediction(short_csv)
        self.assertIn("fewer than 2 points", str(ctx.exception))

    def test_load_all_nan_coordinates(self):
        """File with all NaN coordinates raises ValueError."""
        nan_csv = self.temp_path / "nan.csv"
        df = pd.DataFrame({
            "timestamp_s": [100.0, 101.0],
            "latitude_deg": [np.nan, np.nan],
            "longitude_deg": [np.nan, np.nan],
            "speed_ms": [10.0, 10.0],
            "heading_deg": [90.0, 90.0],
        })
        df.to_csv(nan_csv, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_cpp_prediction(nan_csv)
        self.assertIn("all-NaN coordinate values", str(ctx.exception))

    def test_validate_against_manifest_success(self):
        """Matching prediction passes manifest bounds validation."""
        pred_df = load_cpp_prediction(self.valid_csv)
        manifest_row = {
            "outage_id": "test_01",
            "start_s": 100.0,
            "end_s": 110.0,
            "outage_s": 10.0,
        }
        res = validate_prediction_against_manifest(pred_df, manifest_row)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["issues"]), 0)
        self.assertEqual(res["n_rows"], 100)

    def test_validate_against_manifest_start_and_duration_mismatches(self):
        """Mismatches in start time or duration are detected and reported."""
        pred_df = load_cpp_prediction(self.valid_csv)
        # Manifest expecting start at 90.0s and 30s duration
        manifest_row = {
            "outage_id": "test_mismatch",
            "start_s": 90.0,
            "end_s": 120.0,
            "outage_s": 30.0,
        }
        res = validate_prediction_against_manifest(pred_df, manifest_row)
        self.assertFalse(res["valid"])
        self.assertGreaterEqual(len(res["issues"]), 2)
        issue_text = " ".join(res["issues"])
        self.assertIn("Start timestamp mismatch", issue_text)
        self.assertIn("Duration mismatch", issue_text)


if __name__ == "__main__":
    unittest.main()
