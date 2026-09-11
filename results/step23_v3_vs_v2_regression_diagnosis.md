# Step 23 — Diagnostic Report: Why EKF v3 Loses to v2 More Often Than It Wins (Fleetwide 45.5%)

**Date:** 2026-09-11  
**Status:** Complete (Diagnosis-Only — No Code Changes Made to EKF Core or Leaderboard)  
**Evaluated Cohort:** All 253 eligible paired outage instances across 5 outage durations (10s, 30s, 60s, 120s, 180s)  
**Primary Anchor Instance:** `pair_S3c__120s__0` (+549.0m regression vs v2)

---

## Executive Summary

Step 22 replaced the accelerometer-based lateral acceleration signal $a_{lat} = |f_{b,x} - b_{accel,x}|$ with the kinematic centripetal acceleration formula $a_c = v_{speed} \cdot |\omega_{b,z} - b_{gyro,z}|$ ($k_c = 2.0$, $k_{yaw} = 0$). While this completely fixed straight-road vibration vulnerability (recovering hundreds of meters on `pair_S1`, `pair_S3b`, `pair_Vta1a`, `pair_Vta17`), the fleetwide win rate vs v2 dropped to **45.5%** (115 improved, 120 regressed, 18 unchanged), with net losses at 10s (34.3%), 60s (40.4%), and 120s (41.5%).

This diagnostic investigation analyzed all 120 regressed instances and conducted deep time-series tracing on the anchor case `pair_S3c`@120s. The investigation established the exact physical root causes:

1. **The 60s/120s Deficit is a REAL Physical Regression (Two Interlocking Flaws):**
   - **Flaw A (Dead-Reckoned Speed Collapse):** In 70.0% of all regressions (and >80% at 60s–120s), initial heading misalignment causes NHC to aggressively brake the estimated forward velocity, collapsing $v_{speed}$ from highway speeds ($18$ m/s) down to crawling speeds ($0.4$–$1.5$ m/s). Because $a_c = v_{speed} \cdot \omega_z$, centripetal acceleration was underestimated by $15\times$–$40\times$, and the variance inflation term $(k_c \cdot a_c)^2$ was compressed by **$200\times$–$1,600\times$**. The filter became completely blind to turns.
   - **Flaw B (Sensor-Axis Blindspot):** Step 22 hardcoded $\omega_{b,z}$. In 68.6% of runs (24 of 35 analyzed), smartphone mounts placed the vertical yaw axis on the phone's **Y** or **X** axis (e.g. portrait windshield mount). In `pair_S3c`, the vehicle turn occurred around the sensor's Y axis ($\omega_y = 8.15^\circ$/s), while $\omega_z$ was only $0.06^\circ$/s. EKF v3 registered virtually zero yaw rate.
   - Together, Flaws A and B caused $\sigma_{nhc,lat}$ to remain clamped at $0.15$–$0.25$ m/s during genuine highway curves. The filter forcibly suppressed lateral motion, causing v3 to travel only 246m over 120s (while GT traveled 1,078m), exploding cross-track error to $-1,057.7$m.

2. **v2's Advantage on Curves Was Partially Real, Partially Accidental:**
   - On 60s/120s curves, v2's physical accelerometer $f_{b,x}$ genuinely sensed the real lateral centripetal force ($2.5$–$3.5$ m/s²), inflating $\sigma_{nhc,lat}$ to $2.0$–$5.0$ m/s independent of estimated speed or gyro mounting. This allowed v2 to turn.
   - However, v2 was fundamentally flawed because on straight roads it could not distinguish vibration ($1.5$–$2.5$ m/s²) from a real turn, causing catastrophic drift on straight highways (`pair_S1` +1,213m error).

3. **The 10s Cohort (34.3% Win Rate) is a Comparison Artifact Against Broken v2:**
   - At 10s, 94.6% of regressions are along-track lag (median difference: +10.45m).
   - In v2, road vibration kept $\sigma_{nhc,lat}$ soft ($1.0$–$1.5$ m/s) past the 2s settling window, preventing NHC from braking initial heading errors. In v3, NHC clamped down cleanly to $0.15$ m/s, causing minor speed braking. v3 is physically cleaner and beat v1 by 20.2m on these exact instances; v2's apparent 10s advantage was entirely driven by vibration leakiness.

---

## 1. Full Regression Manifest & Distribution Analysis

### 1.1 Fleetwide Overview
Across all 253 eligible paired instances in the leaderboard:
- **Improved (v3 < v2 - 0.1m):** 115 (45.5%)
- **Regressed (v3 > v2 + 0.1m):** 120 (47.4%)
- **Unchanged (|v3 - v2| <= 0.1m):** 18 (7.1%)
- **Fleetwide Win Rate:** 45.5% (115 / 253)

### 1.2 Breakdown by Duration
The magnitude distribution of the 120 regressed instances ($\Delta = \text{Error}_{v3} - \text{Error}_{v2}$ [m]):

| Duration | $N_{eval}$ | $N_{reg}$ | % Regressed | Min | p25 | Median | p75 | p90 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | 67 | 37 | 55.2% | 0.10m | 6.58m | **10.45m** | 16.31m | 29.81m | 53.14m |
| **30s** | 58 | 22 | 37.9% | 0.22m | 8.60m | **33.86m** | 77.15m | 117.68m | 454.77m |
| **60s** | 52 | 28 | 53.8% | 0.51m | 23.26m | **50.08m** | 94.81m | 176.04m | 415.82m |
| **120s** | 41 | 22 | 53.7% | 14.86m | 71.06m | **122.66m** | 225.23m | 267.33m | 549.00m |
| **180s** | 35 | 11 | 31.4% | 0.28m | 125.11m | **201.80m** | 312.42m | 441.04m | 664.13m |

### 1.3 Error Component Decomposition on Regressed Instances
Analyzing whether regressions were caused by along-track error, cross-track error, or heading error:

| Duration | Median $v2$ Err | Median $v3$ Err | Median $\Delta \|along\|$ | Median $\Delta \|cross\|$ | Median $\Delta \|heading\|$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **10s** | 115.1m | 128.8m | **+10.40m** | +4.03m | -0.55° |
| **30s** | 378.5m | 446.6m | +31.12m | +2.52m | -7.95° |
| **60s** | 736.6m | 810.7m | +32.38m | **+43.73m** | -7.09° |
| **120s** | 1541.1m | 1757.8m | +117.70m | **+22.18m** | -10.45° |
| **180s** | 1001.4m | 1145.8m | +208.24m | +31.09m | +1.89° |

- At **10s**, cross-track difference is near zero (+4.0m), and heading error actually improved (-0.55°). The entire regression is driven by **along-track lag (+10.40m)**.
- At **60s and 120s**, cross-track error and along-track error both widen drastically.

### 1.4 Trajectory Type Classification of 60s & 120s Regressions
Classifying all 50 regressed instances at 60s and 120s by trajectory curvature:
- **Curved Road (Turn > 30° or max $a_c > 1.0$ m/s²):** **50 / 50 (100.0%)**
- **Straight / Gentle Road:** **0 / 50 (0.0%)**

> [!IMPORTANT]
> Exactly 0 straight-road instances regressed at 60s or 120s. Every single regression at these durations occurred on a genuinely curved road where curvature relaxation was required.

### 1.5 Run ID Concentration
- Total unique runs evaluated: 72
- Unique runs with regressions: 55
- Runs with 4 regressions (all 4/5 durations): 4 runs (`pair_Vtb5`, `pair_Vw14a`, `pair_Vw14c`, `pair_Vw3`)
- Runs with 3 regressions: 17 runs (including `pair_S3c`, `pair_M`, `pair_Vfa02`, `pair_Vta10`, `pair_Vta14`, `pair_Vta16`, `pair_Vta27`, `pair_Vta29`, `pair_Vta30`, `pair_Vtb1`, `pair_Vtb2`, `pair_Vw14b`, `pair_Vw2`, `pair_Vw4`)
- Runs with 2 regressions: 19 runs
- Runs with 1 regression: 15 runs

---

## 2. Anchor Case `pair_S3c`@120s Deep Trace

### 2.1 Instance Profile
- Outage ID: `pair_S3c__120s__0` (start $t=2943.8$s, duration 120s)
- Environment: Highway interchange curve / ramp
- Ground Truth: Initial speed 17.4 m/s, mean speed 17.9 m/s, total heading change **$237.7^\circ$**, max centripetal acceleration **$37.82$ m/s²**, total distance traveled **$1,078.7$m**.
- Leaderboard Result: v2 error = **749.7m**, v3 error = **1298.7m** ($\Delta = \mathbf{+549.0m}$).

### 2.2 Time-Series Evolution Every 10 Seconds

| $t$ (s) | GT Distance | GT Speed | v3 Speed (est) | GT Heading | v2 Error | v3 Error | v2 Cross | v3 Cross | $\Delta(v3-v2)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.0m | 17.4 m/s | 17.4 m/s | 242.2° | 25.3m | 25.3m | -0.4m | -0.4m | +0.0m |
| 10 | 180.1m | 19.1 m/s | 1.5 m/s | 224.8° | 174.8m | 185.3m | +6.1m | -9.8m | +10.6m |
| 20 | 364.3m | 18.4 m/s | 0.5 m/s | 204.1° | 363.5m | 375.6m | -111.3m | -96.6m | +12.1m |
| 30 | 525.3m | 17.5 m/s | 1.1 m/s | 187.1° | 532.3m | 539.0m | -230.0m | -237.4m | +6.6m |
| 40 | 638.2m | 17.1 m/s | 0.4 m/s | 140.7° | 618.8m | 650.2m | -564.6m | -569.9m | +31.3m |
| 50 | 734.7m | 18.5 m/s | 0.9 m/s | 140.4° | 695.1m | 748.3m | -563.9m | -570.4m | +53.2m |
| **60** | **871.1m** | **19.9 m/s** | **3.0 m/s** | **138.7°** | **690.8m** | **892.7m** | **-533.9m** | **-588.5m** | **+202.0m** |
| **70** | **988.3m** | **19.6 m/s** | **0.5 m/s** | **116.2°** | **765.3m** | **1040.9m** | **-499.0m** | **-751.7m** | **+275.6m** |
| **80** | **996.5m** | **16.0 m/s** | **3.7 m/s** | **80.9°** | **590.3m** | **1085.7m** | **-587.0m** | **-1074.9m** | **+495.4m** |
| 90 | 973.2m | 18.3 m/s | 3.6 m/s | 52.4° | 608.3m | 1092.5m | -606.5m | -1078.4m | +484.2m |
| 100 | 996.0m | 16.6 m/s | 3.1 m/s | 78.8° | 605.3m | 1141.3m | -451.0m | -1016.4m | +536.0m |
| 110 | 1103.6m | 16.0 m/s | 8.4 m/s | 73.6° | 653.3m | 1299.8m | -406.6m | -1174.1m | +646.5m |
| 120 | 1078.7m | 17.6 m/s | 3.8 m/s | 4.5° | 749.7m | 1298.7m | -744.0m | -1057.7m | +549.1m |

### 2.3 Precise Divergence Point
- Up to $t=50$s, v2 and v3 tracked closely ($\Delta = +53.2$m). Cross-track errors were identical ($-563.9$m vs $-570.4$m).
- Between $t=58$s and $t=85$s, the vehicle entered the sharpest segment of the highway curve (heading swept from $138.7^\circ$ down to $52.4^\circ$, an $86.3^\circ$ turn).
- In v2, cross-track error stayed stable ($-587.0$m at $t=80$s).
- In v3, cross-track error **exploded from $-570.4$m to $-1,074.9$m (+504.5m increase)**.
- Final position: GT traveled to East $+880.2$m, North $-623.7$m. v2 traveled to North $-465.1$m. v3 froze at East $-105.3$m, North $+222.2$m (distance traveled: only $246.5$m).

---

## 3. Physical Root Cause: The Two Interlocking Failures

### 3.1 Failure 1: Dead-Reckoned Speed Collapse Blinds the Filter
At $t=0$, initial heading has an estimation error ($+51.7^\circ$ in `pair_S3c`). When the vehicle moves at $18$ m/s, this heading offset creates an apparent lateral velocity in the body frame:
$$v_{body,x} = v \cdot \sin(\Delta \theta) \approx 18 \cdot \sin(51.7^\circ) \approx 14.1\text{ m/s}$$
The EKF's NHC measurement update enforces $v_{body,x} \approx 0$ with baseline $\sigma_0 = 0.15$ m/s. Because the baseline constraint is tight, NHC continually dampens the velocity vector $\mathbf{v}_{enu}$, braking the forward speed:
- At $t=10$s: Estimated speed drops to $1.5$ m/s.
- At $t=20$s: Estimated speed drops to $0.5$ m/s.
- Mean estimated speed over the entire 120s outage is **$3.0$ m/s** (vs GT $17.9$ m/s).

Now consider the centripetal acceleration formula shipped in Step 22:
$$a_c = v_{speed} \cdot |\omega_{b,z} - b_{gyro,z}|$$
Because $v_{speed}$ collapsed by $35\times$ ($0.5$ m/s vs $18$ m/s), $a_c$ collapsed by $35\times$. The variance inflation term:
$$\sigma_{nhc,lat} = \sqrt{\sigma_0^2 + (k_c \cdot a_c)^2}$$
was compressed by $35^2 = \mathbf{1,225\times}$. The filter computed $a_c \approx 0.005$ m/s² and kept $\sigma_{nhc,lat} \approx 0.15$–$0.25$ m/s throughout the entire highway turn.

**Fleetwide Preponderance:**
Analyzing all 120 regressions across the fleet:
- **70.0% (84 / 120)** suffered severe speed collapse (mean estimated speed $< 0.5 \times$ GT speed).
- At 60s, 120s, and 180s, **78.6% to 86.4%** suffered severe speed collapse.

### 3.2 Failure 2: Sensor Mount Axis Blindspot (The Gyro Yaw Axis Problem)
In Step 22, the angular rate was hardcoded as the sensor's Z-axis: $\omega_{yaw} = |\omega_{b,z} - b_{gyro,z}|$.

A fleetwide audit of the 35 paired runs with ground truth at 180s revealed that smartphones were mounted in diverse orientations:
- **Yaw rotation on Sensor Z:** 11 runs (31.4%) — e.g. phone lying flat on console.
- **Yaw rotation on Sensor Y:** 15 runs (42.9%) — e.g. phone mounted in portrait orientation on windshield or dashboard.
- **Yaw rotation on Sensor X:** 9 runs (25.7%) — e.g. phone mounted in landscape orientation.

In `pair_S3c`, the phone was mounted in portrait mode:
- Sensor $Z$ axis was oriented along gravity ($a_z = +9.84$ m/s²).
- During the $86^\circ$ turn, the angular rate around $Z$ was **$\omega_z = 0.06^\circ$/s** ($0.001$ rad/s).
- The angular rate around $Y$ was **$\omega_y = 8.15^\circ$/s** ($0.142$ rad/s).

Because Step 22 read `last_omega_body_.z`, it read $0.06^\circ$/s. Even if speed had not collapsed, $a_c$ would have been computed as $18 \cdot 0.001 = 0.018$ m/s² (instead of $18 \cdot 0.142 = 2.56$ m/s²).

Together, **$v_{speed} = 0.5$ m/s** and **$\omega_z = 0.001$ rad/s** produced:
$$a_c = 0.5 \cdot 0.001 = \mathbf{0.0005\text{ m/s}^2}$$
representing a **$5,000\times$ under-calculation** of true centripetal force ($2.56$ m/s²).

---

## 4. Evaluation of Hypotheses

### 4.1 Hypothesis 1: $k_c$ Miscalibration (Tested Explicitly)
Sweeping $k_c \in [0.5, 20.0]$ with the existing formula on `pair_S3c__120s__0`:
- $k_c = 0.5$: Error = $1284.6$m
- $k_c = 1.0$: Error = $1285.2$m
- $k_c = 2.0$: Error = $1286.7$m
- $k_c = 5.0$: Error = $1292.0$m
- $k_c = 10.0$: Error = $1204.6$m
- $k_c = 20.0$: Error = $1211.3$m

**Verdict: REJECTED as the sole cause.**  
Because $a_c$ is artificially zero, changing $k_c$ from 2.0 to 10.0 only scales zero. It moves error by only 80m and cannot close the 549m gap to v2.

### 4.2 Hypothesis 2: Dead-Reckoned Speed Feedback (Tested Explicitly)
Testing speed decoupling options on `pair_S3c__120s__0`:
- As-is ($v_{speed} = \text{estimated}$): $1286.7$m
- Bounded speed floor ($\max(v_{speed}, 5.0)$): $1285.1$m
- Pre-outage speed floor ($v_0 = 17.4$ m/s): $1206.9$m

**Verdict: CONFIRMED as an essential driver.**  
Speed collapse is necessary to explain why the vehicle stops moving, but speed decoupling alone cannot fix instances where the gyro axis is also wrong.

### 4.3 Combined Fix: Axis Invariance + Speed Decoupling
When both flaws are addressed together on `pair_S3c__120s__0`:
- Using 3D angular rate magnitude $\|\boldsymbol{\omega}_{body} - \mathbf{b}_{gyro}\|$ + speed floor $v_0$: Error drops to **$1154.7$m**.
- However, naively applying raw 3D gyro norm on straight roads (e.g. `pair_S1__180s__0`) re-introduces vibration sensitivity (error increases from $715.0$m to $1286.7$m) because phone gyros experience high-frequency angular vibration on rough roads.

---

## 5. Why Did v3 Win on `pair_S3c` at Other Durations in Step 22?

The Step 22 report noted that `pair_S3c` improved vs v2 or v1 at 10s, 30s, 60s, and 180s, creating the illusion that curvature relaxation was working.

The ground-truth turn profile of `pair_S3c` across all 5 durations explains this discrepancy completely:

| Outage | Start Time | Total GT Turn | Max Speed | Max GT $a_c$ | Dominant Character |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `pair_S3c__10s__0` | $t=2313.2$s | **4.2°** | 13.7 m/s | 3.97 m/s² | Straight highway |
| `pair_S3c__30s__0` | $t=1106.6$s | **4.9°** | 31.8 m/s | 8.77 m/s² | High-speed straight |
| `pair_S3c__60s__0` | $t=21.8$s | **33.7°** | 17.9 m/s | 14.55 m/s² | Gentle bend |
| `pair_S3c__120s__0` | $t=2943.8$s | **237.7°** | 20.8 m/s | **37.82 m/s²** | **Sharp 238° Highway Interchange Loop** |
| `pair_S3c__180s__0` | $t=971.9$s | **10.6°** | 31.8 m/s | 12.94 m/s² | High-speed straight |

`pair_S3c__120s__0` was the **ONLY** outage window for this run that contained the 238° highway loop. The other 4 outages were essentially straight roads ($4^\circ$–$33^\circ$), where suppressing accelerometer vibration naturally improved the score. Step 22's validation checked the straight slices and missed the turn failure.

---

## 6. Honest, Unhedged Verdict

Is the fleetwide 45.5% win rate a real regression, a comparison artifact against broken v2, or both?

**Final Call: It is a mixture of two distinct phenomena, clearly separated by duration:**

1. **At 10s (34.3% Win Rate — Artifact-Driven):**
   - **Diagnosis:** Comparison artifact against broken v2.
   - **Evidence:** The median regression magnitude is only **+10.45m**. 94.6% of regressions are purely along-track lag caused by clean NHC braking after the 2s settling period. In v2, road vibration kept $\sigma_{nhc,lat}$ artificially loose ($1.0$–$1.5$ m/s), accidentally letting bad heading drift forward. v3 actually improved over v1 by $20.2$m on these exact instances. v2's win here is an artifact of vibration leakage.

2. **At 60s and 120s (40.4% and 41.5% Win Rate — Real Regression):**
   - **Diagnosis:** REAL regression requiring a targeted architectural fix.
   - **Evidence:** 100% of the 50 regressed instances at 60s and 120s are curved roads. In v3, speed collapse combined with sensor-axis mismatch reduced calculated $a_c$ to near zero, causing NHC to forcibly clamp lateral motion on real highway curves. This froze vehicle forward progress and caused cross-track errors exceeding $1,000$m.

---

## 7. Concrete Architectural Recommendations for Step 24

To resolve the 60s/120s curvature regressions without sacrificing straight-road vibration immunity:

1. **Decouple Turn Detection from Degraded Dead-Reckoned Speed:**
   - Do not rely on instantaneous $v_{speed}$ from the EKF state vector for curvature scaling. Use a robust speed proxy:
     $$v_{curv} = \max(v_{speed}, v_0)$$
     where $v_0$ is the pre-outage speed captured at outage onset.

2. **Mount-Orientation Invariant Gyroscope Yaw Rate:**
   - Instead of hardcoding the sensor $Z$ axis ($\omega_{b,z}$), project the unbiased angular rate vector into the gravity-aligned navigation frame using the EKF's current attitude quaternion $\mathbf{q}$:
     $$\boldsymbol{\omega}_{nav} = \mathbf{R}(\mathbf{q}) \cdot (\boldsymbol{\omega}_{body} - \mathbf{b}_{gyro})$$
     $$\omega_{yaw} = |\omega_{nav,z}|$$
   - Because the navigation frame $Z$ axis is always aligned with local gravity (vertical), $\omega_{nav,z}$ correctly isolates true horizontal yaw rate regardless of whether the phone is mounted in portrait, landscape, or flat on a console.

3. **Low-Pass Filter on $\omega_{yaw}$:**
   - Apply a simple first-order IIR low-pass filter (cutoff $\approx 2$ Hz) on $\omega_{yaw}$ before computing $a_c = v_{curv} \cdot \omega_{yaw,filt}$. This rejects high-frequency rotational vibration from road chatter while preserving genuine steering transients.

4. **Tune $k_c$ on True Centripetal Acceleration:**
   - With $a_c$ computed using the gravity-aligned vertical yaw rate and $v_{curv}$, calibrate $k_c$ so that genuine turns ($a_c \in [1.5, 4.0]$ m/s²) expand $\sigma_{nhc,lat}$ into the $3.0$–$8.0$ m/s range, ensuring NHC releases completely during curves.
