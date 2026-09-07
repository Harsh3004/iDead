"""Pair discovery module for matching CAN and Smartphone runs from the synchronised subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd


@dataclass
class PairEntry:
    """Metadata representing a candidate or confirmed CAN/Smartphone paired drive."""

    pair_id: str
    base_id: str
    can_run_id: str
    phone_run_id: str
    can_parquet_path: Optional[Path]
    phone_parquet_path: Optional[Path]
    source_quality_flags: List[str] = field(default_factory=list)
    status: str = "ready"  # "ready", "missing_can", "missing_phone", "failed_source"
    status_detail: str = ""


def _extract_base_id(run_id: str) -> str:
    """Extract common base run ID from vehicle CAN or smartphone run IDs (e.g. 'V-S1' -> 'S1')."""
    r = run_id.strip()
    if r.startswith("V-"):
        return r[2:]
    if r.startswith("S-"):
        return r[2:]
    return r


def _has_sync_path(canonical_path: str, duplicate_paths: str) -> bool:
    """Check whether a run was sourced from or duplicated within the Synchronised dataset folder."""
    combined = f"{canonical_path}|{duplicate_paths}".replace("\\", "/").lower()
    for part in combined.split("/"):
        if "synchronised" in part and "unsynchronised" not in part:
            return True
    return False


def discover_pairs(
    can_manifest_path: Path,
    phone_manifest_path: Path,
    processed_root: Optional[Path] = None,
) -> List[PairEntry]:
    """Discover all paired CAN/Smartphone runs from the Step 3 and Step 4 manifests.

    Args:
        can_manifest_path: Path to data/processed/vehicle_can/manifest.csv.
        phone_manifest_path: Path to data/processed/smartphone/manifest.csv.
        processed_root: Optional root directory containing processed data. If None, derived from manifests.

    Returns:
        Sorted list of PairEntry objects covering all expected and discovered pairs.
    """
    if not can_manifest_path.exists():
        raise FileNotFoundError(f"CAN manifest not found: {can_manifest_path}")
    if not phone_manifest_path.exists():
        raise FileNotFoundError(f"Smartphone manifest not found: {phone_manifest_path}")

    can_m = pd.read_csv(can_manifest_path)
    phone_m = pd.read_csv(phone_manifest_path)

    if processed_root is None:
        processed_root = can_manifest_path.parent.parent

    can_dir = processed_root / "vehicle_can"
    phone_dir = processed_root / "smartphone"

    # Index CAN runs by base_id
    can_by_base: Dict[str, dict] = {}
    for _, row in can_m.iterrows():
        base = _extract_base_id(str(row["run_id"]))
        is_sync = _has_sync_path(
            str(row.get("canonical_source_path", "")),
            str(row.get("duplicate_source_paths", "")),
        )
        can_by_base[base] = {
            "run_id": str(row["run_id"]),
            "is_sync": is_sync,
            "parse_status": str(row.get("parse_status", "unknown")),
            "flags": str(row.get("quality_flags", "none")),
        }

    # Index Phone runs by base_id
    phone_by_base: Dict[str, dict] = {}
    for _, row in phone_m.iterrows():
        base = _extract_base_id(str(row["run_id"]))
        is_sync = _has_sync_path(
            str(row.get("canonical_source_path", "")),
            str(row.get("duplicate_source_paths", "")),
        )
        phone_by_base[base] = {
            "run_id": str(row["run_id"]),
            "is_sync": is_sync,
            "parse_status": str(row.get("parse_status", "unknown")),
            "flags": str(row.get("quality_flags", "none")),
        }

    # Collect all base IDs that appear in Synchronised subset in either or both
    sync_bases: Set[str] = set()
    for base, data in can_by_base.items():
        if data["is_sync"]:
            sync_bases.add(base)
    for base, data in phone_by_base.items():
        if data["is_sync"]:
            sync_bases.add(base)

    entries: List[PairEntry] = []
    for base in sorted(sync_bases):
        pair_id = f"pair_{base}"
        can_info = can_by_base.get(base)
        phone_info = phone_by_base.get(base)

        can_run_id = can_info["run_id"] if can_info else f"V-{base}"
        phone_run_id = phone_info["run_id"] if phone_info else f"S-{base}"

        can_p = can_dir / f"{can_run_id}.parquet"
        phone_p = phone_dir / f"{phone_run_id}.parquet"

        flags: List[str] = []
        if can_info and can_info["flags"] and can_info["flags"] != "none":
            for f in can_info["flags"].split("|"):
                if f and f != "none":
                    flags.append(f"can:{f}")
        if phone_info and phone_info["flags"] and phone_info["flags"] != "none":
            for f in phone_info["flags"].split("|"):
                if f and f != "none":
                    flags.append(f"phone:{f}")

        # Check availability
        if can_info is None:
            entries.append(
                PairEntry(
                    pair_id=pair_id,
                    base_id=base,
                    can_run_id=can_run_id,
                    phone_run_id=phone_run_id,
                    can_parquet_path=None,
                    phone_parquet_path=phone_p if phone_p.exists() else None,
                    source_quality_flags=flags,
                    status="missing_can",
                    status_detail="CAN recording not present in vehicle_can manifest",
                )
            )
            continue

        if phone_info is None:
            entries.append(
                PairEntry(
                    pair_id=pair_id,
                    base_id=base,
                    can_run_id=can_run_id,
                    phone_run_id=phone_run_id,
                    can_parquet_path=can_p if can_p.exists() else None,
                    phone_parquet_path=None,
                    source_quality_flags=flags,
                    status="missing_phone",
                    status_detail="Smartphone recording not present in smartphone manifest",
                )
            )
            continue

        if can_info["parse_status"] != "ok":
            entries.append(
                PairEntry(
                    pair_id=pair_id,
                    base_id=base,
                    can_run_id=can_run_id,
                    phone_run_id=phone_run_id,
                    can_parquet_path=can_p if can_p.exists() else None,
                    phone_parquet_path=phone_p if phone_p.exists() else None,
                    source_quality_flags=flags,
                    status="failed_source",
                    status_detail=f"CAN run parse failed with status {can_info['parse_status']}",
                )
            )
            continue

        if phone_info["parse_status"] != "ok":
            entries.append(
                PairEntry(
                    pair_id=pair_id,
                    base_id=base,
                    can_run_id=can_run_id,
                    phone_run_id=phone_run_id,
                    can_parquet_path=can_p if can_p.exists() else None,
                    phone_parquet_path=phone_p if phone_p.exists() else None,
                    source_quality_flags=flags,
                    status="failed_source",
                    status_detail=f"Phone run parse failed with status {phone_info['parse_status']}",
                )
            )
            continue

        if not can_p.exists():
            entries.append(
                PairEntry(
                    pair_id=pair_id,
                    base_id=base,
                    can_run_id=can_run_id,
                    phone_run_id=phone_run_id,
                    can_parquet_path=None,
                    phone_parquet_path=phone_p if phone_p.exists() else None,
                    source_quality_flags=flags,
                    status="missing_can",
                    status_detail=f"CAN Parquet file missing: {can_p.name}",
                )
            )
            continue

        if not phone_p.exists():
            entries.append(
                PairEntry(
                    pair_id=pair_id,
                    base_id=base,
                    can_run_id=can_run_id,
                    phone_run_id=phone_run_id,
                    can_parquet_path=can_p,
                    phone_parquet_path=None,
                    source_quality_flags=flags,
                    status="missing_phone",
                    status_detail=f"Phone Parquet file missing: {phone_p.name}",
                )
            )
            continue

        entries.append(
            PairEntry(
                pair_id=pair_id,
                base_id=base,
                can_run_id=can_run_id,
                phone_run_id=phone_run_id,
                can_parquet_path=can_p,
                phone_parquet_path=phone_p,
                source_quality_flags=flags,
                status="ready",
                status_detail="",
            )
        )

    return entries
