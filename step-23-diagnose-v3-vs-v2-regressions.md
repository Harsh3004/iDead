# Step 23 — Diagnose Why v3 Loses to v2 More Often Than It Wins (Fleetwide 45.5%)

## Context (what's real on disk right now — read before starting)

- Step 22 shipped `ekf_zupt_nhc_v3` (kinematic centripetal acceleration `a_c = v_speed * |omega_body_z - b_gyro_z|` replacing the accelerometer-based `a_lat`, with `k_c=2.0`, `k_yaw=0` dropped). Full 253-instance batch complete, leaderboard at 1,766 rows.
- **This step exists because the central metric Step 22 was judged on moved the wrong way.** Straight-road vibration immunity is confirmed and real (`pair_S1`, `pair_S3b`, `pair_Vta1a`, `pair_Vta17` all recover hundreds of meters vs v2). But the fleetwide win rate this fix was supposed to improve went from 53.4% (v2 vs v1) to **45.5% (v3 vs v2, 115 improved / 120 regressed / 18 unchanged)** — v2 now beats v3 more often than not.
- Duration breakdown of v3-vs-v2 win rate: 10s 34.3%, 30s 55.2%, 60s 40.4%, 120s 41.5%, 180s 62.9%. The 60s/120s buckets are both net-losing and their **median error also got worse** in absolute terms (60s +55.2m, 120s +59.1m vs v2).
- `pair_S3c` — the instance Step 22 used to validate that genuine turn relaxation survived the sensor swap — regressed **+549.0m at exactly the 120s duration** (749.7m in v2 → 1298.7m in v3), while improving at the other four durations (10s -8.2 vs v1 direction aside, 30s/60s/180s all improved vs v2 or v1). This is the one number in the Step 22 report that doesn't fit the "fix worked" narrative and wasn't explained.
- The 10s win rate vs v2 (34.3%) is also inconsistent with the report's claim that the settling period (untouched) makes 10s "unchanged" — that claim was validated only against corrected-strapdown, not against v2, and the two are not the same comparison.
- Hypothesis space to investigate, not assumed:
  1. **`k_c` miscalibration**: the gyro-derived `a_c` may have a different effective magnitude/timing profile than the old accelerometer `a_lat` on real turns, so `k_c=2.0` (picked to match `pair_S3c`'s aggregate behavior) may over- or under-relax at specific points in a turn where v2's noisier signal happened to align better by coincidence.
  2. **Dead-reckoned speed feedback**: `v_speed` comes from the EKF's own estimate, which itself degrades over the outage. At 60-120s, accumulated speed error could be large enough to distort `a_c` in ways the Step 22 Section 7 analysis (which only checked steady-state/short-horizon behavior) didn't catch.
  3. **Timing/lag mismatch**: gyro-Z and accelerometer may respond to a turn's onset/exit at different apparent times (sensor placement, filtering, bias correction lag) such that curvature relaxation now engages/disengages at a different phase of the turn than v2's did, even if magnitude is right.
  4. Something else entirely — don't stop at the first plausible story; check what the actual per-instance regression list has in common.

## Objective

Identify the actual, evidenced root cause of the 120 fleetwide regressions (v3 vs v2), with `pair_S3c`@120s as the primary anchor case, and determine whether it's fixable with a `k_c`/timing adjustment, requires a different signal formulation, or reveals that v2's apparent 60-120s advantage was itself coincidental (e.g. driven by the same vibration contamination happening to cancel out a different error in a subset of instances — check this before assuming v3 is simply wrong).

## In scope

### 1. Full regression manifest, not just the anchor case
- Pull the complete list of the 120 (v3 vs v2) regressed instances from the leaderboard, broken out by duration and by magnitude of regression (median loss on regressed was reported as 10.45m/33.86m/50.08m/122.66m/201.80m per duration — get the actual distribution, not just the median).
- Check for common structure: are regressions concentrated in specific runs (repeated across durations for the same `pair_id`), specific road/vehicle cohorts, specific speed ranges, or specific curvature magnitudes? Report what's actually there.

### 2. `pair_S3c`@120s deep trace
- Pull the actual `a_c` (v3) and `a_lat` (v2, if recoverable/re-computable) time series for this specific instance's outage window, alongside ground-truth curvature (from gyro-Z/speed directly, independent of either filter's estimate) and the filter's own `sigma_nhc_lat` over time for both v2 and v3.
- Identify precisely where in the 120s window the two versions diverge and what the state estimator was doing differently at that point (NHC gain, heading correction magnitude, whether ZUPT/settling was active).
- Check hypothesis 2 directly: plot the EKF's dead-reckoned `v_speed` error (vs. ground-truth speed) over the 120s window for this instance — is there a period where it's badly wrong, and does that period coincide with the divergence point?

### 3. Test hypothesis 1 (k_c calibration) explicitly
- Recompute what `a_c` and `a_lat` values actually occurred during `pair_S3c`'s real turns (not the assumed "1.5-3.0 m/s²" range) across all 5 durations, and check whether a different `k_c` (sweep a small range) closes the 120s gap without reopening the straight-road vibration problem Step 22 just fixed. Report the sweep results, don't just pick a number.

### 4. Sanity-check whether v2's 60-120s numbers were partly a coincidence
- For the specific instances where v3 regresses vs v2 at 60s/120s, check whether v2's better performance there was itself riding on vibration-induced noise inflation that happened to suppress a different error mode (e.g. over-relaxed NHC accidentally reducing sensitivity to a bad heading estimate) rather than genuine curvature handling. This determines whether v3's numbers are "actually worse" or "correctly worse at a metric that was inflated by a bug."

### 5. Honest verdict
- State plainly whether the fleetwide 45.5% is a real regression needing a fix, an artifact of comparing against a partially-broken v2, or a mix — and for which duration buckets which explanation applies. No averaging past it either way.

## Out of scope
- No new leaderboard config/re-run yet — this is diagnosis-only, same discipline as Step 21. If it points to a specific fix (revised `k_c`, timing correction, alternate formulation), that becomes Step 24's implementation spec.
- No changes to settling period, ZUPT, or filter structure.

## Acceptance criteria
- [ ] Full regression manifest pulled and characterized (not just medians) — concentration by run/cohort/speed/curvature reported honestly.
- [ ] `pair_S3c`@120s time-series trace completed, divergence point identified with supporting numbers (not narrative alone).
- [ ] Dead-reckoned speed error checked directly against the divergence point for this instance.
- [ ] `k_c` sweep run and reported, with an explicit recommendation (keep 2.0 / change to X / signal needs a different form) backed by numbers.
- [ ] v2's 60-120s "wins" checked for whether they ride on the same vibration bug Step 21/22 diagnosed, with a plain verdict either way.
- [ ] Straightforward final call: is this a real net regression, a metric artifact, or both depending on duration — stated without hedging.

## Handoff

Report back with the regression manifest, the `pair_S3c` trace, the `k_c` sweep, and your honest verdict on whether v2's 60-120s numbers were partly artifact-driven. Whatever the finding, I'll write Step 24 as the concrete fix (recalibrated `k_c`, timing correction, or a revised signal) grounded in what this diagnostic actually shows — not as a rerun of Step 22's assumptions.
