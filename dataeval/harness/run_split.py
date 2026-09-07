"""CLI entry point for Step 6: Dataset train/validation/test split assignment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from dataeval.harness.split import assign_split, build_split_groups, write_splits_manifest


def run_dataset_split(
    processed_root: Path,
    out_dir: Path,
    seed: int = 42,
) -> int:
    """Execute dataset split assignment and manifest generation.

    Args:
        processed_root: Path to data/processed containing vehicle_can, smartphone, and paired directories.
        out_dir: Destination directory for splits/manifest.csv and summary.md.
        seed: Random seed for deterministic assignment.

    Returns:
        Exit code: 0 on success.
    """
    start_time = time.time()
    print("Starting Dataset Train/Val/Test Split Assignment:")
    print(f"  Processed root: {processed_root.resolve()}")
    print(f"  Output dir:     {out_dir.resolve()}")
    print(f"  Random seed:    {seed}")

    groups_df = build_split_groups(processed_root)
    print(f"Built {len(groups_df)} atomic groups across dataset.")

    split_df = assign_split(groups_df, seed=seed)

    manifest_csv, summary_md = write_splits_manifest(split_df, out_dir)
    elapsed = time.time() - start_time

    total_dur = split_df["total_duration_s"].sum()
    d_train = split_df[split_df["split"] == "train"]["total_duration_s"].sum() / total_dur * 100.0
    d_val = split_df[split_df["split"] == "val"]["total_duration_s"].sum() / total_dur * 100.0
    d_test = split_df[split_df["split"] == "test"]["total_duration_s"].sum() / total_dur * 100.0

    print(f"\nSplit Assignment Complete in {elapsed:.2f}s:")
    print(
        f"Summary: {len(split_df)} groups partitioned | "
        f"Train: {d_train:.1f}% ({len(split_df[split_df['split'] == 'train'])} grps) / "
        f"Val: {d_val:.1f}% ({len(split_df[split_df['split'] == 'val'])} grps) / "
        f"Test: {d_test:.1f}% ({len(split_df[split_df['split'] == 'test'])} grps) | "
        f"Manifest: {manifest_csv}"
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Partition IO-VNBD dataset into train, val, and test splits.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing processed datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Output directory for splits manifest and summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic assignment.",
    )
    args = parser.parse_args()

    return run_dataset_split(
        processed_root=args.processed_root,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
