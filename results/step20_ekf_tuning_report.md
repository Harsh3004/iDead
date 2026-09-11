# Step 20: EKF Tuning Report — Short-Horizon Settling & Curvature-Adaptive NHC

- **Execution Timestamp:** 2026-09-11 06:28:07Z
- **C++ Replay Wall-Clock Time:** 34.21 s across 253 instances (135.2 ms/instance)
- **Leaderboard Total Rows:** 1513 (377 CV + 377 Bare + 253 Corrected + 253 EKF v1 + 253 EKF v2 = 1,513 rows)

## 1. Executive Summary & Root-Cause Diagnosis

In Step 19, the 15-state ES-EKF demonstrated massive multi-minute gains (-97.2% error reduction at 180s), but two targeted regressions were diagnosed:
1. **10s Outages Regressed vs Corrected Strapdown** (median 94.5m -> 106.4m, only 33/67 [49.3%] improved).
   - **Diagnosis:** Detailed time-series inspection of regressed instances (`pair_Vw16b`, `pair_VfA02`) revealed that state covariance converges within 1-2s and is *not* slowly shrinking. Instead, rigid NHC (sigma=0.15 m/s) applied immediately from t=0 treats normal lateral body velocities (1-3 m/s due to road crown, tire slip, or minor yaw misalignment) as 10-20 sigma violations. This acts as an aggressive braking and steering shock, cutting longitudinal velocity from 23 m/s down to 0.77 m/s within 2 seconds. When NHC was disabled on 10s runs, error dropped back to 95.0m, exactly matching corrected strapdown.
   - **Fix:** An exponential settling grace period on NHC noise: `sigma_nhc(t) = sigma_0 + sigma_settle * exp(-(t - t0) / tau)` (with sigma_settle=4.0 m/s, tau=2.0s). This allows the filter to smoothly settle without initial shock, without needing to know outage duration.
2. **Highway Sustained Curvature Over-Constraint** (`pair_S3c` regressed by +1,118.1m at 180s).
   - **Diagnosis:** During sustained highway curves with lateral acceleration 1.5 - 3.0 m/s^2, rigid lateral NHC (sigma=0.15 m/s) violates real vehicle tire slip angle and roll dynamics, accumulating a false heading correction.
   - **Fix:** Causal curvature-adaptive measurement noise inflation: `sigma_nhc,lat = (sigma_0 + sigma_settle(t)) * sqrt(1 + (k_lat * |a_lat|)^2 + (k_yaw * |omega_yaw|)^2)` (with k_lat=5.0, k_yaw=2.0). When driving straight, a_lat ~ 0 and omega_yaw ~ 0, so sigma_nhc,lat remains strictly 0.15 m/s.

## 2. Five-Way Median & p95 Performance Comparison (Paired Ground-Truth N=253)

| Outage | N | CV Med (m) | Bare Med (m) | Corr Med (m) | EKF v1 Med (m) | EKF v2 Med (m) | EKF v2 p95 (m) | Delta vs Corr (m) | Imp vs Corr | Delta vs v1 (m) | Imp vs v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10s | 67 | 34.1 | 87.1 | 94.5 | 106.4 | 96.3 | 374.7 | +1.8 | -1.9% | -10.1 | +9.5% |
| 30s | 58 | 144.4 | 1037.2 | 1106.1 | 318.8 | 327.4 | 784.0 | -778.7 | +70.4% | +8.7 | -2.7% |
| 60s | 52 | 364.9 | 5368.7 | 5389.9 | 669.0 | 701.4 | 1480.1 | -4688.6 | +87.0% | +32.3 | -4.8% |
| 120s | 41 | 754.3 | 26025.0 | 26351.2 | 1101.1 | 1016.7 | 2837.1 | -25334.5 | +96.1% | -84.4 | +7.7% |
| 180s | 35 | 1142.3 | 47929.0 | 46519.9 | 1300.1 | 1488.9 | 4783.2 | -45031.1 | +96.8% | +188.7 | -14.5% |

## 3. 10s Short-Horizon Outage Deep Dive

| Metric | EKF v1 (`ekf_zupt_nhc_v1`) | EKF v2 (`ekf_zupt_nhc_v2`) | Change |
|---|---|---|---|
| **10s Median Error** | 106.42 m | 96.33 m | **-10.09 m** (+9.5%) |
| **10s Wins vs Corrected** | 33 / 67 (49.3%) | 37 / 67 (55.2%) | **+4 instances** (flipped to majority win) |
| **10s Direct v2 vs v1** | Baseline | 46 improved, 14 regressed | **68.7% improved vs v1** |

## 4. `pair_S3c` High-Speed Curvature Deep Dive

| Outage ID | Duration | Bare (m) | Corrected (m) | EKF v1 (m) | EKF v2 (m) | Delta v2 vs v1 (m) | Delta v2 vs Corr (m) |
|---|---|---|---|---|---|---|---|
| `pair_S3c__10s__0` | 10s | 68.8 | 52.3 | 129.2 | 113.3 | **-15.9** | +61.0 |
| `pair_S3c__30s__0` | 30s | 67.5 | 216.6 | 1006.3 | 910.6 | **-95.7** | +694.0 |
| `pair_S3c__60s__0` | 60s | 2839.5 | 1533.0 | 193.4 | 209.8 | **+16.4** | -1323.2 |
| `pair_S3c__120s__0` | 120s | 41190.2 | 35299.8 | 1279.2 | 749.7 | **-529.4** | -34550.1 |
| `pair_S3c__180s__0` | 180s | 26742.1 | 3941.5 | 5059.6 | 3998.1 | **-1061.4** | +56.7 |

## 5. Side-Effect Audit Across 30s–180s Durations

| Outage | N | v2 Improved vs v1 | v2 Regressed vs v1 | v2 Unchanged | Net Median Shift vs v1 (m) |
|---|---|---|---|---|---|
| 10s | 67 | 46 | 14 | 7 | -10.09 m |
| 30s | 58 | 27 | 27 | 4 | +8.66 m |
| 60s | 52 | 28 | 21 | 3 | +32.34 m |
| 120s | 41 | 20 | 19 | 2 | -84.36 m |
| 180s | 35 | 14 | 19 | 2 | +188.73 m |

### Overall Fleetwide Impact (EKF v2 vs EKF v1, N=253):
- **Improved:** 135 / 253 (53.4%)
- **Regressed:** 100 / 253 (39.5%)
- **Unchanged (within 0.1m):** 18 / 253 (7.1%)
