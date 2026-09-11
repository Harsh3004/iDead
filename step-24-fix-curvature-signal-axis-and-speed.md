# Step 24 — Fix Kinematic Curvature Signal: Mount-Orientation Invariance + Speed-Collapse Decoupling

## Context (what's real on disk right now — read before starting)

- Step 22 shipped `ekf_zupt_nhc_v3`, replacing v2's accelerometer-based curvature signal with `a_c = v_speed * |omega_body_z - b_gyro_z|`. This fully fixed straight-road vibration false-triggering (confirmed: `pair_S1`, `pair_S3b`, `pair_Vta1a` all recovered hundreds of meters), but the central metric — fleetwide win rate vs v2 — came in at **45.5% (115 improved / 120 regressed / 18 unchanged)**, a net loss.
- Step 23 (diagnosis-only, no code changed) traced this to **two interlocking, precisely identified flaws in the v3 formula itself**, not a fundamental problem with the kinematic-signal approach:
  1. **Dead-reckoned speed collapse blinds the detector.** Heading misalignment at outage onset causes NHC to brake the EKF's own forward-speed estimate (observed: 17.4 m/s → 0.4–1.5 m/s on `pair_S3c`@120s). Since `a_c` multiplies by this same collapsed `v_speed`, the centripetal signal was suppressed by up to 1,225× exactly when a real turn was happening. Present in **70% of all 120 regressions**, 78.6–86.4% at 60–180s.
  2. **Sensor-axis blindspot.** The formula hardcodes `omega_body_z`. A fleetwide audit of 35 runs found only 31.4% of phone mounts actually put the yaw axis on sensor Z — 42.9% were Y (portrait) and 25.7% were X (landscape). On `pair_S3c`, true yaw rate was on Y (`omega_y = 8.15°/s`) while `omega_z` read `0.06°/s` — a ~135× miss on that axis alone, compounding with flaw 1.
  - Combined effect on the anchor case `pair_S3c__120s__0`: v3 registered `a_c ≈ 0.0005 m/s²` against a true value of `2.56 m/s²` (5,000× under-calculation), froze the vehicle near-stationary during a real 238° highway loop, and lost 549m to v2.
- Step 23 explicitly tested and **rejected** re-tuning `k_c` alone as a fix (sweeping 0.5–20.0 moved the anchor-case error by only ~80m because it's scaling an already-near-zero signal). It **confirmed** that combining a speed floor with an axis-independent angular rate closes most of the gap on the anchor case (1298.7m → 1154.7m), and separately flagged that a naive 3D gyro-norm reintroduces vibration sensitivity on straight roads (must not skip the low-pass step).
- Step 23's regression analysis also cleanly separated the 10s cohort (34.3% win rate) as **not a real problem** — it's along-track lag from v3 correctly clamping NHC where v2's vibration leakage accidentally let bad initial heading drift forward; v3 already beats v1 there by 20.2m. **Do not chase the 10s number in this step** — changes here should be judged on 60–180s.
- Leaderboard has 1,766 rows across configs `baseline_cv_heading_v1`, bare/corrected strapdown, `ekf_zupt_nhc_v1`, `ekf_zupt_nhc_v2`, `ekf_zupt_nhc_v3`. This step adds a fifth: `ekf_zupt_nhc_v4`. None of the prior four are to be modified or deleted.

## Objective

Implement Step 23's three concrete recommendations — pre-outage speed floor, gravity-aligned (mount-orientation-invariant) yaw rate, and a low-pass filter to reject rotational road vibration — as `ekf_zupt_nhc_v4`. Validate specifically against the exact failure this step exists to fix (curved-road speed collapse + off-axis mounts) using the anchor case and the fleetwide regression set Step 23 already identified, without regressing the straight-road vibration immunity v3 established.

## In scope

### 1. Pre-outage speed floor (decouple `a_c` from degraded dead-reckoned speed)
- At outage onset, capture and hold `v_0` = the EKF's forward-speed estimate at the last GNSS-healthy fix before the outage begins.
- Use `v_curv = max(v_speed, v_0)` (not a straight substitution) as the speed term in the centripetal formula, so a genuine mid-outage slowdown (real braking) is not masked, but NHC-driven collapse toward zero cannot suppress the signal below what the vehicle was actually doing at outage start.
- Justify the `max()` choice vs. alternatives (e.g., decaying floor, blend) with the actual before/after numbers on the anchor case and on any regressed instance involving genuine deceleration — check that this floor doesn't now over-inflate curvature on a real, legitimate slow-down-then-stop scenario.

### 2. Mount-orientation-invariant yaw rate
- Replace `omega_body_z` with the gravity-aligned navigation-frame yaw rate: `omega_nav = R(q) * (omega_body - b_gyro)`, `omega_yaw = |omega_nav_z|`, using the EKF's current attitude quaternion — same rotation math already used elsewhere in the filter, no new state.
- Confirm this is invariant across portrait/landscape/flat mounts by re-running the axis audit from Step 23 (or a synthetic equivalent) and showing `omega_yaw` now tracks true horizontal turn rate regardless of which body axis it landed on.

### 3. Low-pass filter on `omega_yaw`
- Apply a first-order IIR low-pass (cutoff ≈ 2 Hz, or justify a different cutoff with data) to `omega_yaw` before computing `a_c = v_curv * omega_yaw_filt`.
- Verify this is necessary and sufficient: confirm on a known vibration-heavy straight instance (`pair_S1` or similar) that raw gravity-aligned `omega_yaw` alone reintroduces false inflation (Step 23 flagged this risk when testing the 3D gyro norm), and that the filtered version suppresses it back toward v3's clean straight-road behavior.

### 4. Re-tune `k_c` on the corrected signal
- With `a_c` now computed from true centripetal motion (not an artificially suppressed one), recalibrate `k_c` so genuine turns (`a_c` in the 1.5–4.0 m/s² range, per Step 23's recommendation) expand `sigma_nhc_lat` into roughly 3.0–8.0 m/s, sufficient for NHC to fully release during curves. Show the sweep and justify the final value with numbers, not assertion.

### 5. Targeted validation before the full batch
- **Anchor case first:** confirm `pair_S3c__120s__0` no longer freezes — report its full error and the time-series trace (same 10s-interval table Step 23 produced) showing the vehicle actually completes the 238° turn instead of traveling only 246m.
- **Regression set:** re-run at minimum the 50 curved-road 60s/120s instances Step 23 identified as 100% curved and currently regressed, and report how many now beat v2.
- **Straight-road set:** re-confirm zero regression on the vibration-sensitive instances v3 already fixed (`pair_S1`, `pair_S3b`, `pair_Vta1a`, `pair_Vta17`) — these must not come back.
- Only proceed to the full 253-instance batch once both directions check out; if either fails, iterate on the filter/floor/k_c before committing leaderboard rows.

### 6. Full-scale re-run and leaderboard update
- Run the full 253-instance batch as `ekf_zupt_nhc_v4`, append to `results/leaderboard.csv`, verify all 1,766 prior rows byte-identical.
- Report the same three comparisons Steps 20/22 did: (a) v4 vs. corrected-strapdown baseline per duration, (b) v4 vs. v3 per duration + fleetwide win rate (this is the number Step 23 diagnosed — report it explicitly, especially the 60s/120s breakdown since that's where the real problem was), (c) v4 vs. v2 and v4 vs. v1 fleetwide, so we can see whether v4 is now a clean superset of both prior architectures' strengths.
- Report the 10s cohort for completeness but do not treat it as pass/fail criteria per the note above.

### 7. Tests
- Extend the C++/Python suites: a synthetic test with the phone mounted in each of 3 orientations (yaw on X, Y, Z) producing identical `sigma_nhc_lat` output for the same physical turn — this is the exact axis-invariance test that would have caught Step 22's bug.
- A synthetic test combining heading-misalignment-induced speed collapse with a real sustained turn, confirming `a_c` stays representative of the true turn rather than collapsing with the corrupted speed estimate.
- Keep Step 22's existing vibration-immunity and turn-detection tests passing.

## Out of scope
- No changes to ZUPT, the settling grace period, or any other EKF subsystem untouched by Steps 20–23.
- No changes to `ekf_zupt_nhc_v1`/`v2`/`v3` code paths or leaderboard rows — all three remain as comparison history.
- Do not attempt to fix or explain the 10s cohort's win rate — it's already understood and is not a regression.

## Acceptance criteria
- [ ] Pre-outage speed floor implemented (`v_curv = max(v_speed, v_0)`), with `v_0` captured correctly at outage onset in both C++ and Python.
- [ ] Gravity-aligned `omega_yaw` implemented, confirmed axis-invariant across mount orientations with real numbers.
- [ ] Low-pass filter on `omega_yaw` implemented and confirmed necessary (raw signal reintroduces vibration sensitivity; filtered signal doesn't).
- [ ] `k_c` re-tuned on the corrected signal with sweep data shown.
- [ ] `pair_S3c__120s__0` anchor case no longer freezes; full trace reported showing turn completion.
- [ ] Fleetwide 60s/120s curved-road regression set (the 50 instances from Step 23) re-checked; improvement count reported.
- [ ] Straight-road vibration immunity (`pair_S1`, `pair_S3b`, `pair_Vta1a`, `pair_Vta17`) confirmed still intact — no regression from v3.
- [ ] Full 253-instance batch run as `ekf_zupt_nhc_v4`, leaderboard grows correctly, prior 1,766 rows byte-identical.
- [ ] Fleetwide v4-vs-v3 win rate reported, with 60s/120s broken out explicitly (this is what's actually being fixed).
- [ ] v4-vs-v2, v4-vs-v1, v4-vs-corrected-strapdown comparisons reported.
- [ ] New axis-invariance and speed-collapse-with-real-turn tests pass alongside existing suite.
- [ ] Honest reporting of any new failure mode this combination introduces — in particular, check whether the speed floor causes over-inflation on any instance with a genuine, legitimate slowdown during a turn (e.g., approaching a stop sign mid-curve), since that's the one scenario Step 23 didn't get to test.

## Handoff

Implement all three fixes together (they're interlocking, per Step 23's diagnosis — none alone closes the gap), validate against the anchor case and both regression/immunity sets before running the full batch, then report the fleetwide v4-vs-v3 win rate and whether the 60s/120s deficit is actually closed. If it is, `ekf_zupt_nhc_v4` becomes the fleet's real Phase 1 baseline and we move to whatever's next (wider validation on unpaired instances, or Android/C++ hardening). If a new failure mode surfaces — especially the legitimate-slowdown-during-turn case — report it plainly with numbers and I'll write the next diagnostic step.
