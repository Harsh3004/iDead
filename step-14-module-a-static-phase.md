# Step 14 — Module A v1: Static-Phase Attitude Init (Gravity → Pitch/Roll, Gyro Bias)

## Context (what's real on disk right now — read before starting)

- Phase 0 is closed. You have: 90 CAN Parquet + 97 phone Parquet (Steps 3–4),
  72 paired 10 Hz runs (Step 5), a train/val/test split (Step 6), 378 outage
  instances (Step 7), a scoring harness with two baselines on the leaderboard
  (Steps 8/10), and a green CI pipeline (Step 12).
- **The critical finding from Step 10 that this step exists to start fixing:**
  bare strapdown integration with **zero initial-attitude correction and zero
  gyro-bias correction** loses to naive constant-velocity by 2.6×–42.0×,
  growing roughly as `t^3` — exactly the signature of uncorrected gyro bias
  (confirmed synthetically in Step 9: 60s/30s error ratio = 7.999 vs
  theoretical 8.0). Module A does not fix the outage-window drift itself
  (that's the EKF/ZUPT's job in Step 19) — it fixes the **inputs** the
  strapdown engine starts each outage with: a correct initial roll/pitch and
  a subtracted-out constant gyro bias, both estimated from data the vehicle
  already generates before it starts moving.
- **This is a Python-only step.** All estimation/ML work in Phase 1 is
  prototyped in `dataeval` first (as Step 8's baseline was) — do not touch
  the C++ core (`core/`) yet. Porting Module A into C++ is Phase 3
  productisation work (Step 28+), not now.
- **Exact column names are not fully known to me for phone IMU fields** — my
  context has the paired schema at 57 columns (`can_*`/`phone_*` prefixed)
  and the phone canonical schema at 27 columns, but not the literal
  accelerometer/gyroscope field names. **Before writing any estimation code,
  inspect one paired Parquet file's actual column list and report it** — do
  not guess names or assume axis conventions (e.g. phone-frame vs.
  vehicle-frame, deg/s vs rad/s, m/s² vs g). Get this from the data, not from
  memory of the raw IO-VNBD column headers.

## Objective

Build a static-phase attitude/bias initializer: for a given run, find a
genuinely stationary segment, and from it estimate (a) initial roll and
pitch from the measured gravity vector, and (b) a constant gyro bias vector
(one estimate per axis) from the raw gyro readings during that segment.
Validate it against synthetic ground truth and sanity-check it against real
runs. **Do not wire this into the C++ strapdown replay or the scoring
harness yet** — that requires Step 15's yaw estimate too, so a full initial
attitude quaternion is available in one place. This step's job is: build the
static-phase estimator correctly, prove it works on synthetic data, and
characterize what it says about the real dataset.

## In scope

### 1. Stationary-segment detection
- For each of the 72 paired runs (they have both CAN wheel-speed and phone
  IMU, so bias estimates can be cross-checked against known-zero speed),
  find a genuinely stationary window: near-zero wheel speed AND near-zero
  GPS speed AND low accelerometer variance, sustained for at least a few
  seconds. Runs typically have such a window at the very start (before the
  vehicle pulls away) — check this assumption against real data rather than
  assuming it always exists.
- Pick a principled, stated threshold (e.g. speed < some cm/s, accel std
  below some g-threshold over a rolling window) — don't hand-tune per run.
  Report how many of the 72 paired runs have a usable stationary window at
  all, and how long the found windows are (min/median/max duration).
- Handle the two runs already flagged `zero_wheel_speed`/parked
  (`V-Vw1`, `V-Vw15` from Step 3) sensibly — they may be entirely
  stationary, which changes what "the static window" means for them; call
  this out rather than silently mishandling it.
- Flag (don't silently drop) any run where no stationary window can be found
  with your threshold — Module A simply has no static-phase estimate for
  that run, and downstream steps need to know that explicitly.

### 2. Gravity → pitch/roll
- Average the accelerometer vector over the detected stationary window
  (averaging suppresses sensor noise; a stationary IMU should read
  approximately `-g` along the true vertical in its own frame).
- Derive pitch and roll from that averaged vector using the standard
  leveling equations (confirm/state your axis convention explicitly — don't
  assume; get it from column names/units found in step 1's schema
  inspection). Output units: radians, documented in code and report.
- Sanity bound: a road vehicle at rest should have small pitch/roll (a few
  degrees at most, barring a very steep driveway). Report the actual
  distribution across all runs with a usable window — flag any outliers for
  review rather than treating them as certainly wrong.

### 3. Gyro bias estimation
- Average the raw gyro readings over the same stationary window on all
  three axes. A truly stationary IMU should read ~0 rad/s (Earth's rotation
  rate, ~15°/hr, is far below consumer MEMS noise floor — ignore it, note
  that you did).
- Report the estimated bias per axis per run: distribution (min/median/max,
  in deg/s for human readability). Sanity-check against Step 9's synthetic
  gyro-bias test scale (it used some bias magnitude to produce the `t^3`
  demo) — are the real estimated biases in a plausible consumer-MEMS range
  (typically hundredths to a few degrees/second)? Flag anything wildly off.

### 4. Synthetic validation (unit tests)
- Construct synthetic stationary IMU data with a **known** injected gravity
  vector (known pitch/roll) and a **known** injected gyro bias, run it
  through the estimator, and assert recovered values match within a stated
  tolerance. This must be a from-scratch synthetic fixture, not a fitted
  real run — the point is proving the *math* is right independent of real
  data quality.
- Add a noisy variant (small Gaussian noise added to the synthetic signal)
  and confirm the averaging still recovers values within a looser, stated
  tolerance — proves the averaging approach is doing its job.
- Add a test for the "no stationary window found" path — confirm it's
  reported/flagged, not silently defaulted to zero.

### 5. Output artifact
- Write one row per run to a new file, e.g.
  `results/module_a_static_phase.csv`, columns: `run_id`, whether a
  stationary window was found, window start/end/duration, `pitch_rad`,
  `roll_rad`, `gyro_bias_x/y/z` (state units in a header comment or docs),
  and any quality flag (e.g. `short_window`, `high_variance`,
  `no_window_found`).
- This is a **new** file — do not touch `results/leaderboard.csv` or
  `results/leaderboard_summary.csv` in this step.

## Out of scope
- No yaw estimation (Step 15 — needs PCA on dynamic motion + GNSS
  cross-check, fundamentally different from a stationary-window approach).
- No wiring into the C++ strapdown replay, no re-scoring, no leaderboard
  changes — that needs Step 15's yaw first, and belongs in a later
  integration step.
- No C++ port of this logic (Phase 3, Step 28+).
- No per-outage-instance bias re-estimation or continuous bias tracking —
  this step produces **one static estimate per run**, established once
  before the run starts moving. Whether that's sufficient (bias might drift
  over a long trip) is a question for later evaluation, not to solve here.
- No changes to CAN-only runs (Module A needs phone IMU; if CAN-only files
  lack usable gyro/accel, that's simply out of scope for this pass — 72
  paired runs is enough to validate and characterize the approach).

## Acceptance criteria
- [ ] Real paired-Parquet column names/units/axis convention for
      accelerometer and gyroscope fields reported (not assumed).
- [ ] Stationary-window detection threshold stated explicitly; coverage
      reported (how many of 72 runs have a usable window, durations).
- [ ] `V-Vw1`/`V-Vw15` (entirely-stationary runs) handled and reported
      explicitly, not silently mishandled.
- [ ] Pitch/roll and gyro-bias distributions across all runs with a usable
      window reported, with outliers flagged.
- [ ] Synthetic unit tests (clean signal, noisy signal, no-window-found
      path) pass and are added to the suite (report new total vs. 64).
- [ ] `results/module_a_static_phase.csv` written; existing leaderboard
      files untouched.
- [ ] Report back: axis convention/units discovered, window coverage
      numbers, pitch/roll and bias distributions, any flagged outliers, and
      new test count.

## Handoff

Implement this, run it across all 72 paired runs, and report back the real
numbers — especially the axis convention you found (I don't want to assume
it) and how many runs actually have a usable stationary window, since that
determines how much of the dataset Module A can help at all. Once I have
those numbers, I'll write Step 15 (dynamic-phase PCA yaw estimation +
GNSS cross-check) calibrated against what Module A's static phase actually
found.
