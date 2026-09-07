"""Leakage-safe Train/Validation/Test partition assignment harness for IO-VNBD."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


HARD_COHORTS = [
    "phone:low_sample_rate",
    "phone:burst_sampling",
    "phone:no_magnetometer",
    "stationary_run",
    "phone:clock_reset",
    "low_confidence_offset",
]


def get_campaign_group(base_id: str) -> str:
    """Map a run base ID to its indivisible campaign / route-family group ID."""
    b = base_id.strip()
    if b.startswith("S3"):
        return "Driver_A_S3_trip"  # S3a, S3b, S3c are sequential legs of the same evening trip
    elif b in ["S1", "S2", "S4"]:
        return f"Driver_A_{b}"
    elif b == "M":
        return "Driver_B_M"
    elif b.startswith("St"):
        return f"Driver_C_{b}"
    elif b.startswith("Y"):
        return f"Driver_D_{b}"
    elif b.startswith("Vfa"):
        return "Driver_E_Vfa"
    elif b.startswith("Vfb"):
        return "Driver_E_Vfb"
    elif b.startswith("Vta"):
        return "Driver_E_Vta"
    elif b.startswith("Vtb"):
        return "Driver_E_Vtb"
    elif b.startswith("Vw"):
        return "Driver_E_Vw"
    elif b.startswith("T"):
        return f"Driver_F_{b}"
    elif b == "I":
        return "Driver_G_I"
    elif b.startswith("A"):
        return f"Driver_H_{b}"
    return f"Other_{b}"


def build_split_groups(processed_root: Path) -> pd.DataFrame:
    """Build atomic split groups across CAN, Smartphone, and Paired datasets.

    Args:
        processed_root: Path to data/processed containing vehicle_can, smartphone, and paired directories.

    Returns:
        DataFrame with one row per indivisible group.
    """
    can_m_path = processed_root / "vehicle_can" / "manifest.csv"
    phone_m_path = processed_root / "smartphone" / "manifest.csv"
    paired_m_path = processed_root / "paired" / "manifest.csv"

    if not can_m_path.exists():
        raise FileNotFoundError(f"CAN manifest missing: {can_m_path}")
    if not phone_m_path.exists():
        raise FileNotFoundError(f"Smartphone manifest missing: {phone_m_path}")
    if not paired_m_path.exists():
        raise FileNotFoundError(f"Paired manifest missing: {paired_m_path}")

    can_m = pd.read_csv(can_m_path)
    phone_m = pd.read_csv(phone_m_path)
    paired_m = pd.read_csv(paired_m_path)

    paired_bases = set(r["pair_id"].replace("pair_", "") for _, r in paired_m.iterrows())

    items = []

    # 1. Paired runs
    for _, r in paired_m.iterrows():
        b = r["pair_id"].replace("pair_", "")
        grp = get_campaign_group(b)
        dist = float(phone_m[phone_m["run_id"] == r["phone_run_id"]]["distance_km"].iloc[0])
        items.append({
            "type": "paired",
            "id": r["pair_id"],
            "base_id": b,
            "group_id": grp,
            "can_run_id": r["can_run_id"],
            "phone_run_id": r["phone_run_id"],
            "paired_id": r["pair_id"],
            "duration_s": float(r["aligned_duration_s"]),
            "distance_km": dist,
            "flags": str(r.get("quality_flags", "none")),
        })

    # 2. Unpaired CAN runs
    for _, r in can_m.iterrows():
        b = r["run_id"].replace("V-", "")
        if b not in paired_bases:
            grp = get_campaign_group(b)
            raw_flags = str(r.get("quality_flags", "none"))
            flag_str = "|".join(f"can:{f}" for f in raw_flags.split("|") if f and f != "none")
            items.append({
                "type": "can_only",
                "id": r["run_id"],
                "base_id": b,
                "group_id": grp,
                "can_run_id": r["run_id"],
                "phone_run_id": "",
                "paired_id": "",
                "duration_s": float(r["duration_s"]),
                "distance_km": float(r["distance_km"]),
                "flags": flag_str if flag_str else "none",
            })

    # 3. Unpaired Phone runs
    for _, r in phone_m.iterrows():
        b = r["run_id"].replace("S-", "")
        if b not in paired_bases:
            grp = get_campaign_group(b)
            raw_flags = str(r.get("quality_flags", "none"))
            flag_str = "|".join(f"phone:{f}" for f in raw_flags.split("|") if f and f != "none")
            items.append({
                "type": "phone_only",
                "id": r["run_id"],
                "base_id": b,
                "group_id": grp,
                "can_run_id": "",
                "phone_run_id": r["run_id"],
                "paired_id": "",
                "duration_s": float(r["duration_s"]),
                "distance_km": float(r["distance_km"]),
                "flags": flag_str if flag_str else "none",
            })

    df_items = pd.DataFrame(items)

    # Group aggregation
    group_rows = []
    for grp_id, gdf in df_items.groupby("group_id"):
        all_flags = set()
        for f_str in gdf["flags"]:
            for token in f_str.split("|"):
                if token and token != "none":
                    all_flags.add(token)

        can_runs = [x for x in gdf["can_run_id"] if x]
        phone_runs = [x for x in gdf["phone_run_id"] if x]
        paired_runs = [x for x in gdf["paired_id"] if x]
        run_ids = sorted(set(can_runs + phone_runs))

        group_rows.append({
            "group_id": grp_id,
            "n_drives": len(gdf),
            "n_paired": len(paired_runs),
            "paired_duration_s": gdf[gdf["type"] == "paired"]["duration_s"].sum(),
            "run_ids": "|".join(run_ids),
            "can_run_ids": "|".join(sorted(set(can_runs))),
            "phone_run_ids": "|".join(sorted(set(phone_runs))),
            "paired_pair_ids": "|".join(sorted(set(paired_runs))),
            "total_duration_s": float(gdf["duration_s"].sum()),
            "total_distance_km": round(float(gdf["distance_km"].sum()), 3),
            "quality_flags": "|".join(sorted(all_flags)) if all_flags else "none",
        })

    df_groups = pd.DataFrame(group_rows)
    return df_groups.sort_values("group_id").reset_index(drop=True)


def assign_split(
    groups_df: pd.DataFrame,
    seed: int = 42,
    target_train: float = 0.70,
    target_val: float = 0.15,
    target_test: float = 0.15,
) -> pd.DataFrame:
    """Assign groups deterministically to train, val, and test splits.

    Guarantees:
    - No group is divided across splits.
    - Paired runs in a group stay strictly together.
    - Achieves ~70/15/15 ratio by recording duration.
    - Paired data is intentionally over-represented in val/test.
    - Every hard cohort is represented in val or test.

    Args:
        groups_df: DataFrame of atomic groups from build_split_groups.
        seed: Random seed for deterministic assignment optimization.
        target_train: Target fraction of duration for training (default 0.70).
        target_val: Target fraction of duration for validation (default 0.15).
        target_test: Target fraction of duration for testing (default 0.15).

    Returns:
        DataFrame with added 'split' and 'notes' columns.
    """
    df = groups_df.copy()
    durations = df["total_duration_s"].values
    paired_durations = df["paired_duration_s"].values
    total_dur = durations.sum()
    n_groups = len(df)

    # Boolean matrix for hard cohort presence in each group
    cohort_matrix = np.zeros((n_groups, len(HARD_COHORTS)), dtype=bool)
    for j, h in enumerate(HARD_COHORTS):
        for i in range(n_groups):
            if h in df["quality_flags"].iloc[i]:
                cohort_matrix[i, j] = True

    np.random.seed(seed)
    N_TRIALS = 200000

    # Draw assignments: 0 = train, 1 = val, 2 = test
    all_assigns = np.random.choice([0, 1, 2], size=(N_TRIALS, n_groups), p=[target_train, target_val, target_test])

    is_val = (all_assigns == 1)
    is_test = (all_assigns == 2)
    is_train = (all_assigns == 0)

    # Coverage of hard cohorts in val and test
    val_cohorts = (is_val.astype(int) @ cohort_matrix.astype(int)) > 0
    test_cohorts = (is_test.astype(int) @ cohort_matrix.astype(int)) > 0
    val_or_test_cohorts = val_cohorts | test_cohorts
    all_covered = val_or_test_cohorts.all(axis=1)

    valid_indices = np.where(all_covered)[0]
    if len(valid_indices) == 0:
        raise RuntimeError("Optimization failed to find an assignment covering all hard cohorts")

    train_durs = (is_train[valid_indices] * durations).sum(axis=1) / total_dur
    val_durs = (is_val[valid_indices] * durations).sum(axis=1) / total_dur
    test_durs = (is_test[valid_indices] * durations).sum(axis=1) / total_dur

    val_paired = (is_val[valid_indices] * paired_durations).sum(axis=1)
    test_paired = (is_test[valid_indices] * paired_durations).sum(axis=1)

    # Paired runs must be sufficiently present in val and test
    paired_ok = (val_paired >= 3600 * 3.0) & (test_paired >= 3600 * 3.0)

    # Weighted MSE loss towards 70/15/15 target
    losses = (train_durs - target_train)**2 + 2.0 * (val_durs - target_val)**2 + 2.0 * (test_durs - target_test)**2
    both_covered = (val_cohorts[valid_indices] & test_cohorts[valid_indices]).sum(axis=1)
    losses = losses - 0.002 * both_covered
    losses[~paired_ok] = 1e9

    best_idx = valid_indices[np.argmin(losses)]
    best_assign = all_assigns[best_idx]

    split_map = {0: "train", 1: "val", 2: "test"}
    df["split"] = [split_map[a] for a in best_assign]

    # Verify hard invariants
    assert set(df["split"].unique()) == {"train", "val", "test"}
    val_flags = set("|".join(df[df["split"] == "val"]["quality_flags"]).split("|"))
    test_flags = set("|".join(df[df["split"] == "test"]["quality_flags"]).split("|"))
    combined_eval_flags = val_flags | test_flags

    for h in HARD_COHORTS:
        assert h in combined_eval_flags, f"Hard cohort '{h}' missing from both val and test!"

    # Annotate notes
    notes = []
    for _, r in df.iterrows():
        n = []
        if r["n_paired"] > 0:
            n.append(f"{r['n_paired']} paired drives")
        if r["quality_flags"] != "none":
            n.append(f"flags: {r['quality_flags']}")
        notes.append("; ".join(n) if n else "standard nominal run")
    df["notes"] = notes

    return df


def write_splits_manifest(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Write split manifest.csv and summary.md to output directory.

    Args:
        df: Partitioned groups DataFrame.
        out_dir: Destination directory (e.g. data/processed/splits).

    Returns:
        Tuple of (manifest_csv_path, summary_md_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_dir / "manifest.csv"

    # Export canonical columns
    cols = [
        "group_id",
        "split",
        "run_ids",
        "can_run_ids",
        "phone_run_ids",
        "paired_pair_ids",
        "total_duration_s",
        "total_distance_km",
        "quality_flags",
        "notes",
    ]
    df[cols].to_csv(manifest_csv, index=False)

    # Generate summary.md
    summary_md = out_dir / "summary.md"
    total_dur = df["total_duration_s"].sum()
    total_dist = df["total_distance_km"].sum()

    lines = [
        "# IO-VNBD Train / Validation / Test Split Summary",
        "",
        "## Partition Ratios",
        "",
        "| Split | Groups | Drives | Duration (hrs) | Duration % | Distance (km) | Distance % | Paired Runs | Paired Duration % of Split |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        d_hrs = sub["total_duration_s"].sum() / 3600.0
        d_pct = sub["total_duration_s"].sum() / total_dur * 100.0
        dist = sub["total_distance_km"].sum()
        dist_pct = dist / total_dist * 100.0
        n_p = sub["n_paired"].sum()
        p_hrs = sub["paired_duration_s"].sum() / 3600.0
        p_pct_of_split = (p_hrs / d_hrs * 100.0) if d_hrs > 0 else 0.0

        lines.append(
            f"| **{s.upper()}** | {len(sub)} | {sub['n_drives'].sum()} | "
            f"{d_hrs:.2f} | {d_pct:.1f}% | {dist:.1f} | {dist_pct:.1f}% | "
            f"{n_p} | {p_pct_of_split:.1f}% |"
        )

    lines.extend([
        f"| **TOTAL** | {len(df)} | {df['n_drives'].sum()} | "
        f"{total_dur/3600.0:.2f} | 100.0% | {total_dist:.1f} | 100.0% | "
        f"{df['n_paired'].sum()} | {df['paired_duration_s'].sum() / total_dur * 100.0:.1f}% |",
        "",
        "## Hard-Cohort Representation in Evaluation Splits",
        "",
        "| Cohort / Flag | Present in Train | Present in Val | Present in Test | Representative Evaluated Groups |",
        "|:---|:---:|:---:|:---:|:---|",
    ])

    for h in HARD_COHORTS:
        in_train = any(h in f for f in df[df["split"] == "train"]["quality_flags"])
        in_val = any(h in f for f in df[df["split"] == "val"]["quality_flags"])
        in_test = any(h in f for f in df[df["split"] == "test"]["quality_flags"])

        val_grps = df[(df["split"] == "val") & (df["quality_flags"].str.contains(h, na=False))]["group_id"].tolist()
        test_grps = df[(df["split"] == "test") & (df["quality_flags"].str.contains(h, na=False))]["group_id"].tolist()
        rep_str = ", ".join([f"Val: {g}" for g in val_grps] + [f"Test: {g}" for g in test_grps])

        lines.append(
            f"| `{h}` | {'Yes' if in_train else 'No'} | {'Yes' if in_val else 'No'} | "
            f"{'Yes' if in_test else 'No'} | {rep_str} |"
        )

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return manifest_csv, summary_md
