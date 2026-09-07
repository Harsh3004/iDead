"""CLI entry point for Step 7: GNSS Outage Simulation and evaluation dataset generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataeval.harness.outage import (
    DEFAULT_EDGE_MARGIN_S,
    DEFAULT_OUTAGE_DURATIONS,
    OutageSkip,
    OutageWindow,
    OutageWindowList,
    apply_outage,
    select_outage_windows,
)


def discover_split_entities(
    processed_root: Path,
    splits: Sequence[str],
) -> List[Dict[str, any]]:
    """Discover all distinct run and pair entities belonging to the requested splits.

    Args:
        processed_root: Path to data/processed.
        splits: List of split names to include (e.g. ['val', 'test']).

    Returns:
        List of entity dictionaries with keys:
        'id', 'split', 'type' ('paired', 'can', or 'phone'), 'path', 'has_cross_stream_gt'.
    """
    split_manifest_path = processed_root / "splits" / "manifest.csv"
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Step 6 splits manifest not found at {split_manifest_path}")

    split_df = pd.read_csv(split_manifest_path)
    selected_splits = set(splits)

    entities: List[Dict[str, any]] = []
    seen_ids: Set[str] = set()

    for _, row in split_df.iterrows():
        sp = str(row["split"]).strip()
        if sp not in selected_splits:
            continue

        # 1. Paired runs
        pair_ids_str = str(row.get("paired_pair_ids", "") or "")
        paired_in_row: Set[str] = set()
        if pd.notna(row.get("paired_pair_ids")) and pair_ids_str.strip() and pair_ids_str != "nan":
            for pid in pair_ids_str.split("|"):
                pid = pid.strip()
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    paired_in_row.add(pid)
                    parquet_path = processed_root / "paired" / f"{pid}.parquet"
                    entities.append({
                        "id": pid,
                        "split": sp,
                        "type": "paired",
                        "path": parquet_path,
                        "has_cross_stream_gt": True,
                    })

        # 2. Unpaired CAN runs
        can_ids_str = str(row.get("can_run_ids", "") or "")
        if pd.notna(row.get("can_run_ids")) and can_ids_str.strip() and can_ids_str != "nan":
            for cid in can_ids_str.split("|"):
                cid = cid.strip()
                if not cid or cid in seen_ids:
                    continue
                # If this CAN run is part of a paired run in this group, skip it (it's already handled as paired)
                base = cid.replace("V-", "")
                if any(base in p for p in paired_in_row):
                    continue
                seen_ids.add(cid)
                parquet_path = processed_root / "vehicle_can" / f"{cid}.parquet"
                entities.append({
                    "id": cid,
                    "split": sp,
                    "type": "can",
                    "path": parquet_path,
                    "has_cross_stream_gt": False,
                })

        # 3. Unpaired Phone runs
        phone_ids_str = str(row.get("phone_run_ids", "") or "")
        if pd.notna(row.get("phone_run_ids")) and phone_ids_str.strip() and phone_ids_str != "nan":
            for pid in phone_ids_str.split("|"):
                pid = pid.strip()
                if not pid or pid in seen_ids:
                    continue
                base = pid.replace("S-", "")
                if any(base in p for p in paired_in_row):
                    continue
                seen_ids.add(pid)
                parquet_path = processed_root / "smartphone" / f"{pid}.parquet"
                entities.append({
                    "id": pid,
                    "split": sp,
                    "type": "phone",
                    "path": parquet_path,
                    "has_cross_stream_gt": False,
                })

    return entities


def run_outage_simulation(
    processed_root: Path,
    out_dir: Path,
    splits: Sequence[str] = ("val", "test"),
    durations: Sequence[float] = DEFAULT_OUTAGE_DURATIONS,
    side: str = "phone",
    n_per_duration: int = 1,
    seed: int = 42,
    avoid_real_gaps: bool = True,
    avoid_run_edges: bool = True,
    edge_margin_s: float = DEFAULT_EDGE_MARGIN_S,
) -> int:
    """Execute GNSS outage generation across the requested dataset splits.

    Args:
        processed_root: Root directory of processed data (data/processed).
        out_dir: Output directory for outage parquets and manifest (data/processed/outages).
        splits: Splits to evaluate (e.g. ['val', 'test']).
        durations: Outage durations in seconds.
        side: Target stream to blank ('phone', 'can', or 'both').
        n_per_duration: Number of outage instances per duration per entity.
        seed: Random seed for deterministic window placement.
        avoid_real_gaps: Whether to avoid authentic sensor gaps.
        avoid_run_edges: Whether to preserve leading and trailing buffer margins.
        edge_margin_s: Edge margin in seconds.

    Returns:
        Exit code: 0 on success.
    """
    start_time = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    entities = discover_split_entities(processed_root, splits)
    if not entities:
        print(f"No entities found for splits: {splits}")
        return 1

    print(f"Starting GNSS Outage Simulation:")
    print(f"  Processed root:    {processed_root.resolve()}")
    print(f"  Output directory:  {out_dir.resolve()}")
    print(f"  Splits to process: {list(splits)}")
    print(f"  Durations (s):     {[int(d) if d.is_integer() else d for d in durations]}")
    print(f"  Source side:       {side}")
    print(f"  Windows/duration:  {n_per_duration}")
    print(f"  Random seed:       {seed}")
    print(f"  Edge margin:       {edge_margin_s:.1f}s (avoid_edges={avoid_run_edges})")
    print(f"  Avoid real gaps:   {avoid_real_gaps}")
    print(f"  Discovered {len(entities)} unique run/pair entities across {list(splits)}.\n")

    manifest_rows: List[Dict[str, any]] = []
    generated_count = 0
    skipped_count = 0
    gt_count = 0
    nogt_count = 0
    duration_gen_counts: Dict[float, int] = {d: 0 for d in durations}
    skip_reason_counts: Dict[str, int] = {}

    for entity in entities:
        ent_id = entity["id"]
        ent_split = entity["split"]
        ent_type = entity["type"]
        parquet_path = entity["path"]
        has_gt = entity["has_cross_stream_gt"]

        # Check compatibility with requested source_side
        if side == "phone" and ent_type == "can":
            for d in durations:
                reason = "stream_not_present: run is unpaired CAN without phone GNSS channels"
                manifest_rows.append({
                    "outage_id": "",
                    "run_or_pair_id": ent_id,
                    "split": ent_split,
                    "source_side": side,
                    "outage_s": int(d) if float(d).is_integer() else float(d),
                    "start_s": "",
                    "end_s": "",
                    "overlaps_real_gap": False,
                    "has_cross_stream_ground_truth": False,
                    "status": "skipped",
                    "skip_reason": reason,
                })
                skipped_count += 1
                cat = reason.split(":")[0]
                skip_reason_counts[cat] = skip_reason_counts.get(cat, 0) + 1
            continue

        if side == "can" and ent_type == "phone":
            for d in durations:
                reason = "stream_not_present: run is unpaired Phone without CAN GNSS channels"
                manifest_rows.append({
                    "outage_id": "",
                    "run_or_pair_id": ent_id,
                    "split": ent_split,
                    "source_side": side,
                    "outage_s": int(d) if float(d).is_integer() else float(d),
                    "start_s": "",
                    "end_s": "",
                    "overlaps_real_gap": False,
                    "has_cross_stream_ground_truth": False,
                    "status": "skipped",
                    "skip_reason": reason,
                })
                skipped_count += 1
                cat = reason.split(":")[0]
                skip_reason_counts[cat] = skip_reason_counts.get(cat, 0) + 1
            continue

        if not parquet_path.exists():
            for d in durations:
                reason = f"missing_parquet_file: {parquet_path.name} not found"
                manifest_rows.append({
                    "outage_id": "",
                    "run_or_pair_id": ent_id,
                    "split": ent_split,
                    "source_side": side,
                    "outage_s": int(d) if float(d).is_integer() else float(d),
                    "start_s": "",
                    "end_s": "",
                    "overlaps_real_gap": False,
                    "has_cross_stream_ground_truth": has_gt,
                    "status": "skipped",
                    "skip_reason": reason,
                })
                skipped_count += 1
                cat = reason.split(":")[0]
                skip_reason_counts[cat] = skip_reason_counts.get(cat, 0) + 1
            continue

        df = pd.read_parquet(parquet_path)

        res, skips = select_outage_windows(
            df=df,
            run_id=ent_id,
            durations=durations,
            side=side,
            n_per_duration=n_per_duration,
            seed=seed,
            avoid_real_gaps=avoid_real_gaps,
            avoid_run_edges=avoid_run_edges,
            edge_margin_s=edge_margin_s,
            return_skipped=True,
        )

        # Track placed windows by duration to assign sequential index
        placed_indices: Dict[float, int] = {}
        for win in res:
            d_key = win.duration_s
            idx = placed_indices.get(d_key, 0)
            placed_indices[d_key] = idx + 1

            d_str = f"{int(win.duration_s)}" if win.duration_s.is_integer() else f"{win.duration_s}"
            outage_id = f"{ent_id}__{d_str}s__{idx}"

            # Apply outage
            masked_df = apply_outage(df, win)

            # Write masked parquet
            split_dir = out_dir / ent_split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_file = split_dir / f"{outage_id}.parquet"

            table = pa.Table.from_pandas(masked_df, preserve_index=False)
            pq.write_table(table, out_file, compression="snappy")

            manifest_rows.append({
                "outage_id": outage_id,
                "run_or_pair_id": ent_id,
                "split": ent_split,
                "source_side": side,
                "outage_s": int(win.duration_s) if win.duration_s.is_integer() else win.duration_s,
                "start_s": win.start_s,
                "end_s": win.end_s,
                "overlaps_real_gap": win.overlaps_real_gap,
                "has_cross_stream_ground_truth": has_gt,
                "status": "ok",
                "skip_reason": "",
            })
            generated_count += 1
            duration_gen_counts[win.duration_s] = duration_gen_counts.get(win.duration_s, 0) + 1
            if has_gt:
                gt_count += 1
            else:
                nogt_count += 1

        # Record skips
        for sk in skips:
            manifest_rows.append({
                "outage_id": "",
                "run_or_pair_id": ent_id,
                "split": ent_split,
                "source_side": side,
                "outage_s": int(sk.duration_s) if sk.duration_s.is_integer() else sk.duration_s,
                "start_s": "",
                "end_s": "",
                "overlaps_real_gap": False,
                "has_cross_stream_ground_truth": has_gt,
                "status": "skipped",
                "skip_reason": sk.reason,
            })
            skipped_count += 1
            cat = sk.reason.split(":")[0]
            skip_reason_counts[cat] = skip_reason_counts.get(cat, 0) + 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_cols = [
        "outage_id",
        "run_or_pair_id",
        "split",
        "source_side",
        "outage_s",
        "start_s",
        "end_s",
        "overlaps_real_gap",
        "has_cross_stream_ground_truth",
        "status",
        "skip_reason",
    ]
    # Sort deterministically
    manifest_df = manifest_df[manifest_cols].sort_values(
        by=["split", "run_or_pair_id", "outage_s", "status"],
        ascending=[True, True, True, False],
    )
    manifest_csv = out_dir / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    elapsed = time.time() - start_time

    # Construct one-line summary
    dur_summary = ", ".join(f"{int(d)}s: {duration_gen_counts.get(d, 0)}" for d in durations)
    skip_summary = ", ".join(f"{k}: {v}" for k, v in skip_reason_counts.items())

    print("=" * 80)
    print(
        f"OUTAGE SIMULATION COMPLETE in {elapsed:.2f}s | "
        f"Total Instances Generated: {generated_count} ({dur_summary}) | "
        f"Cross-Stream GT: {gt_count} paired, {nogt_count} unpaired | "
        f"Total Skips: {skipped_count} ({skip_summary}) | "
        f"Manifest: {manifest_csv}"
    )
    print("=" * 80)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic GNSS outage datasets for dead reckoning evaluation.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing processed datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/outages"),
        help="Output directory for outage parquets and manifest.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Dataset split(s) to process ('val', 'test', 'train', or 'all').",
    )
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=DEFAULT_OUTAGE_DURATIONS,
        help="Outage durations in seconds (default: 10 30 60 120 180).",
    )
    parser.add_argument(
        "--side",
        choices=["phone", "can", "both"],
        default="phone",
        help="Target stream to blank (default: phone).",
    )
    parser.add_argument(
        "--n-per-duration",
        type=int,
        default=1,
        help="Number of non-overlapping windows per duration per run (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic window placement.",
    )
    parser.add_argument(
        "--allow-real-gap-overlap",
        action="store_true",
        help="Allow synthetic outage windows to overlap authentic recording gaps (default: False).",
    )
    parser.add_argument(
        "--allow-run-edges",
        action="store_true",
        help="Disable edge margin constraint, allowing outages at run boundaries (default: False).",
    )
    parser.add_argument(
        "--edge-margin-s",
        type=float,
        default=DEFAULT_EDGE_MARGIN_S,
        help=f"Edge margin duration in seconds (default: {DEFAULT_EDGE_MARGIN_S}s).",
    )

    args = parser.parse_args()

    # Handle 'all' in splits
    splits = args.splits
    if "all" in splits:
        splits = ["train", "val", "test"]

    return run_outage_simulation(
        processed_root=args.processed_root,
        out_dir=args.out_dir,
        splits=splits,
        durations=args.durations,
        side=args.side,
        n_per_duration=args.n_per_duration,
        seed=args.seed,
        avoid_real_gaps=not args.allow_real_gap_overlap,
        avoid_run_edges=not args.allow_run_edges,
        edge_margin_s=args.edge_margin_s,
    )


if __name__ == "__main__":
    sys.exit(main())
