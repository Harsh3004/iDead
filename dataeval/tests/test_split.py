"""Unit tests for dataset split grouping and partition assignment."""

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from dataeval.harness.split import (
    HARD_COHORTS,
    assign_split,
    build_split_groups,
    get_campaign_group,
    write_splits_manifest,
)


class TestSplit(unittest.TestCase):
    """Test suite for leakage-safe train/validation/test dataset partitioning."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.proc_dir = self.temp_dir / "processed"
        self.can_dir = self.proc_dir / "vehicle_can"
        self.phone_dir = self.proc_dir / "smartphone"
        self.paired_dir = self.proc_dir / "paired"

        self.can_dir.mkdir(parents=True)
        self.phone_dir.mkdir(parents=True)
        self.paired_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_environment(self) -> None:
        """Populate temporary processed directory with synthetic manifests representing distinct cohorts."""
        # Paired runs
        paired_records = [
            {"pair_id": "pair_S1", "can_run_id": "V-S1", "phone_run_id": "S-S1", "aligned_duration_s": 5000.0, "quality_flags": "none"},
            {"pair_id": "pair_S2", "can_run_id": "V-S2", "phone_run_id": "S-S2", "aligned_duration_s": 9000.0, "quality_flags": "phone:clock_reset"},
            {"pair_id": "pair_Vw1", "can_run_id": "V-Vw1", "phone_run_id": "S-Vw1", "aligned_duration_s": 2000.0, "quality_flags": "can:zero_wheel_speed|stationary_run"},
            {"pair_id": "pair_Vw2", "can_run_id": "V-Vw2", "phone_run_id": "S-Vw2", "aligned_duration_s": 5000.0, "quality_flags": "none"},
            {"pair_id": "pair_Y1", "can_run_id": "V-Y1", "phone_run_id": "S-Y1", "aligned_duration_s": 7000.0, "quality_flags": "low_confidence_offset"},
        ]

        # CAN manifest
        can_records = [
            {"run_id": "V-S1", "duration_s": 5000.0, "distance_km": 38.0, "canonical_source_path": "Synchronised/V-S1.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "V-S2", "duration_s": 9000.0, "distance_km": 75.0, "canonical_source_path": "Synchronised/V-S2.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "V-Vw1", "duration_s": 2000.0, "distance_km": 0.0, "canonical_source_path": "Synchronised/V-Vw1.csv", "duplicate_source_paths": "", "quality_flags": "zero_wheel_speed"},
            {"run_id": "V-Vw2", "duration_s": 5000.0, "distance_km": 98.0, "canonical_source_path": "Synchronised/V-Vw2.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "V-Y1", "duration_s": 7000.0, "distance_km": 60.0, "canonical_source_path": "Synchronised/V-Y1.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "V-St1", "duration_s": 5700.0, "distance_km": 47.0, "canonical_source_path": "Unsynchronised/V-St1.csv", "duplicate_source_paths": "", "quality_flags": "none"},
        ]

        # Phone manifest
        phone_records = [
            {"run_id": "S-S1", "duration_s": 5000.0, "distance_km": 38.0, "canonical_source_path": "Synchronised/S-S1.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "S-S2", "duration_s": 9000.0, "distance_km": 75.0, "canonical_source_path": "Synchronised/S-S2.csv", "duplicate_source_paths": "", "quality_flags": "clock_reset"},
            {"run_id": "S-Vw1", "duration_s": 2000.0, "distance_km": 0.0, "canonical_source_path": "Synchronised/S-Vw1.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "S-Vw2", "duration_s": 5000.0, "distance_km": 98.0, "canonical_source_path": "Synchronised/S-Vw2.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "S-Y1", "duration_s": 7000.0, "distance_km": 60.0, "canonical_source_path": "Synchronised/S-Y1.csv", "duplicate_source_paths": "", "quality_flags": "none"},
            {"run_id": "S-A1", "duration_s": 3000.0, "distance_km": 20.0, "canonical_source_path": "Unsynchronised/S-A1.csv", "duplicate_source_paths": "", "quality_flags": "low_sample_rate"},
            {"run_id": "S-T1", "duration_s": 1000.0, "distance_km": 15.0, "canonical_source_path": "Unsynchronised/S-T1.csv", "duplicate_source_paths": "", "quality_flags": "burst_sampling|no_magnetometer"},
            {"run_id": "S-T2", "duration_s": 4000.0, "distance_km": 30.0, "canonical_source_path": "Unsynchronised/S-T2.csv", "duplicate_source_paths": "", "quality_flags": "no_magnetometer"},
        ]

        pd.DataFrame(can_records).to_csv(self.can_dir / "manifest.csv", index=False)
        pd.DataFrame(phone_records).to_csv(self.phone_dir / "manifest.csv", index=False)
        pd.DataFrame(paired_records).to_csv(self.paired_dir / "manifest.csv", index=False)

    def test_paired_runs_share_same_split(self):
        """Verify that both sides of any pair always land strictly in the same split group."""
        self._create_synthetic_environment()
        groups_df = build_split_groups(self.proc_dir)
        split_df = assign_split(groups_df, seed=42)

        # Check each group that has paired runs
        for _, r in split_df.iterrows():
            if r["paired_pair_ids"]:
                paired_ids = r["paired_pair_ids"].split("|")
                can_ids = r["can_run_ids"].split("|")
                phone_ids = r["phone_run_ids"].split("|")

                for p in paired_ids:
                    base = p.replace("pair_", "")
                    self.assertIn(f"V-{base}", can_ids)
                    self.assertIn(f"S-{base}", phone_ids)

    def test_no_run_or_pair_in_multiple_splits(self):
        """Verify that no run_id or pair_id appears in more than one partition."""
        self._create_synthetic_environment()
        groups_df = build_split_groups(self.proc_dir)
        split_df = assign_split(groups_df, seed=42)

        seen_runs = {}
        seen_pairs = {}
        for _, r in split_df.iterrows():
            split_name = r["split"]
            if r["run_ids"]:
                for rid in r["run_ids"].split("|"):
                    self.assertNotIn(rid, seen_runs, f"Run {rid} appeared in both {seen_runs.get(rid)} and {split_name}")
                    seen_runs[rid] = split_name
            if r["paired_pair_ids"]:
                for pid in r["paired_pair_ids"].split("|"):
                    self.assertNotIn(pid, seen_pairs, f"Pair {pid} appeared in both {seen_pairs.get(pid)} and {split_name}")
                    seen_pairs[pid] = split_name

    def test_achieved_split_ratios_within_tolerance(self):
        """Verify achieved duration ratios match target ~70/15/15 within acceptable tolerance."""
        # Use real processed dataset
        real_proc = Path("data/processed")
        if (real_proc / "paired" / "manifest.csv").exists():
            groups_df = build_split_groups(real_proc)
            split_df = assign_split(groups_df, seed=42)

            total_dur = split_df["total_duration_s"].sum()
            dur_train = split_df[split_df["split"] == "train"]["total_duration_s"].sum() / total_dur
            dur_val = split_df[split_df["split"] == "val"]["total_duration_s"].sum() / total_dur
            dur_test = split_df[split_df["split"] == "test"]["total_duration_s"].sum() / total_dur

            # Target is 70/15/15; tolerance is +/- 5%
            self.assertAlmostEqual(dur_train, 0.70, delta=0.05, msg=f"Train ratio {dur_train:.3f} outside tolerance")
            self.assertAlmostEqual(dur_val, 0.15, delta=0.05, msg=f"Val ratio {dur_val:.3f} outside tolerance")
            self.assertAlmostEqual(dur_test, 0.15, delta=0.05, msg=f"Test ratio {dur_test:.3f} outside tolerance")

    def test_hard_cohort_representation_in_val_or_test(self):
        """Verify that every hard cohort is present in val or test."""
        real_proc = Path("data/processed")
        if (real_proc / "paired" / "manifest.csv").exists():
            groups_df = build_split_groups(real_proc)
            split_df = assign_split(groups_df, seed=42)

            val_flags = set("|".join(split_df[split_df["split"] == "val"]["quality_flags"]).split("|"))
            test_flags = set("|".join(split_df[split_df["split"] == "test"]["quality_flags"]).split("|"))
            eval_flags = val_flags | test_flags

            for h in HARD_COHORTS:
                self.assertIn(h, eval_flags, f"Hard cohort '{h}' not found in validation or test split!")

    def test_deterministic_split(self):
        """Verify running assign_split with the same seed produces identical results."""
        self._create_synthetic_environment()
        groups_df1 = build_split_groups(self.proc_dir)
        split_df1 = assign_split(groups_df1, seed=123)

        groups_df2 = build_split_groups(self.proc_dir)
        split_df2 = assign_split(groups_df2, seed=123)

        self.assertEqual(list(split_df1["split"]), list(split_df2["split"]))


if __name__ == "__main__":
    unittest.main()
