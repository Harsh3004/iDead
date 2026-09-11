import pandas as pd
import numpy as np
from pathlib import Path
from dataeval.harness.run_corrected_replay import load_module_b_tiers

tiers = load_module_b_tiers(Path("results/module_b_initial_attitude.csv"))
manifest = pd.read_csv("data/processed/outages/manifest.csv")
eligible = manifest[
    (manifest["status"] == "ok")
    & (manifest["has_cross_stream_ground_truth"] == True)
    & (manifest["run_or_pair_id"].isin(tiers.keys()))
    & (manifest["outage_s"] == 180) # Check longest outage per run to get full turns
].copy()

results = []
for _, row in eligible.iterrows():
    run_id = row["run_or_pair_id"]
    oid = row["outage_id"]
    split = row["split"]
    
    cache_path = Path(f"data/processed/_cpp_replay_cache/{oid}.csv")
    if not cache_path.exists():
        continue
    
    df_cache = pd.read_csv(cache_path, comment="#")
    dt = np.diff(df_cache["timestamp_s"].values)
    
    int_gx = np.degrees(np.sum(df_cache["gx"].values[:-1] * dt))
    int_gy = np.degrees(np.sum(df_cache["gy"].values[:-1] * dt))
    int_gz = np.degrees(np.sum(df_cache["gz"].values[:-1] * dt))
    
    # GT heading change
    parquet_path = Path(f"data/processed/outages/{split}/{oid}.parquet")
    df_parquet = pd.read_parquet(parquet_path)
    from dataeval.harness.outage import OutageWindow, ground_truth_trajectory
    window = OutageWindow(
        run_id=run_id, source_side=row["source_side"],
        start_s=float(row["start_s"]), duration_s=float(row["outage_s"]), end_s=float(row["end_s"])
    )
    gt = ground_truth_trajectory(df_parquet, window)
    gt_hdg = gt["heading_deg"].values
    gt_turn = np.degrees(np.unwrap(np.radians(gt_hdg))[-1] - np.unwrap(np.radians(gt_hdg))[0])
    
    results.append({
        "run_id": run_id,
        "oid": oid,
        "source": row["source_side"],
        "gt_turn": gt_turn,
        "int_gx": int_gx,
        "int_gy": int_gy,
        "int_gz": int_gz,
        "az_mean": df_cache["az"].mean(),
        "ay_mean": df_cache["ay"].mean(),
        "ax_mean": df_cache["ax"].mean(),
    })

res_df = pd.DataFrame(results)
print(f"Analyzed {len(res_df)} runs at 180s:")
for _, r in res_df.iterrows():
    # Identify which axis best matches gt_turn (in abs or inverted)
    err_x = abs(abs(r["int_gx"]) - abs(r["gt_turn"]))
    err_y = abs(abs(r["int_gy"]) - abs(r["gt_turn"]))
    err_z = abs(abs(r["int_gz"]) - abs(r["gt_turn"]))
    best = "Z" if (err_z <= err_x and err_z <= err_y) else ("Y" if err_y <= err_x else "X")
    print(f"{r['run_id']:12s} ({r['source']:5s}): GT turn={r['gt_turn']:7.1f}° | gx={r['int_gx']:7.1f}°, gy={r['int_gy']:7.1f}°, gz={r['int_gz']:7.1f}° | BEST={best} | az={r['az_mean']:5.2f}, ay={r['ay_mean']:5.2f}")

