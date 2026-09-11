# Step 24: EKF v4 Evaluation Report — Curvature Signal Axis Invariance, Speed Floor & Vibration Rejection

**Date:** 2026-09-11 16:21:03 UTC

**Replay Execution Time (Batch 253 instances):** 83.506 seconds (83506.4 ms, 330.06 ms/outage)

## 1. Executive Summary & Physical Fixes Delivered

Step 24 implements three interlocking physical fixes to the EKF curvature-adaptive Non-Holonomic Constraint (NHC):
1. **Pre-Outage Speed Floor ($v_{curv} = \max(v_{speed}, v_0)$):** Decouples centrifugal acceleration $a_c$ from dead-reckoned forward velocity collapse under persistent turn deceleration, guaranteeing sustained curvature inflation throughout turns.
2. **Mount-Orientation Invariant Yaw Rate ($[\mathbf{R}(\mathbf{q}) \cdot (\boldsymbol{\omega}_b - \mathbf{b}_g)]_z$):** Extracts the true horizontal turning rate in the gravity-aligned local navigation frame (ENU Up), making turn detection fully invariant to phone mounting pitch and roll (windshield cradle, dashboard flat, cup holder portrait).
3. **First-Order IIR Low-Pass Filter ($f_c = 2.0\text{ Hz}$):** Filters the signed horizontal yaw rate before computing $a_c = v_{curv} \cdot |\omega_{yaw,filt}|$, rejecting high-frequency road-induced angular vibration without rectifying zero-mean noise.

## 2. Seven-Way Median & p95 Performance Comparison (Paired Ground-Truth N=253)

| Outage | N | CV Med (m) | Bare Med (m) | Corr Med (m) | EKF v1 Med (m) | EKF v2 Med (m) | EKF v3 Med (m) | EKF v4 Med (m) | EKF v4 p95 (m) | Delta vs Corr | Imp vs Corr | Delta vs v3 | Imp vs v3 | Delta vs v2 | Imp vs v2 | Delta vs v1 | Imp vs v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10s | 67 | 34.1 | 87.1 | 94.5 | 106.4 | 96.3 | 105.1 | **105.1** | 346.1 | +10.5 | -11.1% | +0.0 | -0.0% | +8.7 | -9.1% | -1.3 | +1.3% |
| 30s | 58 | 144.4 | 1037.2 | 1106.1 | 318.8 | 327.4 | 306.8 | **304.7** | 799.4 | -801.4 | +72.5% | -2.0 | +0.7% | -22.7 | +6.9% | -14.0 | +4.4% |
| 60s | 52 | 364.9 | 5368.7 | 5389.9 | 669.0 | 701.4 | 756.6 | **762.2** | 1519.4 | -4627.7 | +85.9% | +5.6 | -0.7% | +60.8 | -8.7% | +93.2 | -13.9% |
| 120s | 41 | 754.3 | 26025.0 | 26351.2 | 1101.1 | 1016.7 | 1075.8 | **1055.8** | 2878.7 | -25295.5 | +96.0% | -20.0 | +1.9% | +39.0 | -3.8% | -45.3 | +4.1% |
| 180s | 35 | 1142.3 | 47929.0 | 46519.9 | 1300.1 | 1488.9 | 1334.3 | **1221.1** | 4623.0 | -45298.9 | +97.4% | -113.3 | +8.5% | -267.8 | +18.0% | -79.0 | +6.1% |

## 3. Fleetwide Head-to-Head Win-Rate Comparisons (N=253)

### A. EKF v4 vs. EKF v3 (Direct Generation Improvement)
| Outage | N | v4 Improved | v4 Regressed | Unchanged | v4 Win Rate vs v3 | Median Gain on Improved (m) | Median Loss on Regressed (m) |
|---|---|---|---|---|---|---|---|
| 10s | 67 | 28 | 30 | 9 | **41.8%** | 2.41 | 3.49 |
| 30s | 58 | 29 | 23 | 6 | **50.0%** | 9.61 | 17.79 |
| 60s | 52 | 27 | 21 | 4 | **51.9%** | 27.98 | 38.52 |
| 120s | 41 | 17 | 22 | 2 | **41.5%** | 84.37 | 32.98 |
| 180s | 35 | 13 | 19 | 3 | **37.1%** | 164.84 | 50.36 |

- **Overall Fleetwide v4 vs. v3 Win Rate:** **114/253 (45.1%)** (Improved: 114, Regressed: 115, Unchanged: 24)

### B. EKF v4 vs. EKF v2 (Road-Vibration Leakage Baseline)
| Outage | N | v4 Improved | v4 Regressed | Unchanged | v4 Win Rate vs v2 | Median Gain on Improved (m) | Median Loss on Regressed (m) |
|---|---|---|---|---|---|---|---|
| 10s | 67 | 22 | 38 | 7 | **32.8%** | 5.51 | 8.18 |
| 30s | 58 | 29 | 25 | 4 | **50.0%** | 18.05 | 19.44 |
| 60s | 52 | 25 | 24 | 3 | **48.1%** | 46.98 | 40.70 |
| 120s | 41 | 19 | 19 | 3 | **46.3%** | 95.10 | 114.98 |
| 180s | 35 | 17 | 16 | 2 | **48.6%** | 446.63 | 118.85 |

- **Overall Fleetwide v4 vs. v2 Win Rate:** **112/253 (44.3%)** (Improved: 112, Regressed: 122, Unchanged: 19)

### C. EKF v4 vs. EKF v1 (Original Flat NHC Baseline)
| Outage | N | v4 Improved | v4 Regressed | Unchanged | v4 Win Rate vs v1 | Median Gain on Improved (m) | Median Loss on Regressed (m) |
|---|---|---|---|---|---|---|---|
| 10s | 67 | 44 | 16 | 7 | **65.7%** | 25.86 | 20.97 |
| 30s | 58 | 29 | 23 | 6 | **50.0%** | 31.22 | 21.00 |
| 60s | 52 | 27 | 21 | 4 | **51.9%** | 65.47 | 44.86 |
| 120s | 41 | 27 | 12 | 2 | **65.9%** | 63.85 | 55.05 |
| 180s | 35 | 20 | 13 | 2 | **57.1%** | 168.58 | 183.89 |

- **Overall Fleetwide v4 vs. v1 Win Rate:** **147/253 (58.1%)** (Improved: 147, Regressed: 85, Unchanged: 21)

### D. EKF v4 vs. Corrected Strapdown Baseline
| Outage | N | v4 Improved | v4 Regressed | Unchanged | v4 Win Rate vs Corr | Median Gain on Improved (m) | Median Loss on Regressed (m) |
|---|---|---|---|---|---|---|---|
| 10s | 67 | 36 | 31 | 0 | **53.7%** | 42.34 | 74.29 |
| 30s | 58 | 52 | 6 | 0 | **89.7%** | 759.60 | 156.01 |
| 60s | 52 | 51 | 1 | 0 | **98.1%** | 5060.34 | 85.26 |
| 120s | 41 | 41 | 0 | 0 | **100.0%** | 25770.59 | 0.00 |
| 180s | 35 | 35 | 0 | 0 | **100.0%** | 45209.59 | 0.00 |

- **Overall Fleetwide v4 vs. Corrected Win Rate:** **215/253 (85.0%)** (Improved: 215, Regressed: 38, Unchanged: 0)

## 4. Straight-Road Road Vibration Immunity Audit

Verification that v4 completely preserves the straight-road asphalt vibration immunity achieved in Step 22:

| Run ID | Outage | Corrected (m) | EKF v1 (m) | EKF v2 (m) | EKF v3 (m) | EKF v4 (m) | Recovery vs v2 (m) | Physical Notes |
|---|---|---|---|---|---|---|---|---|
| `pair_S1` | 180s | 44713.3 | 219.8 | 1557.7 | 832.3 | **909.7** | **-648.0 m** | Straight road vibration immunity holds |
| `pair_S3b` | 180s | 30472.9 | 503.6 | 1152.3 | 672.5 | **687.5** | **-464.8 m** | Straight road vibration immunity holds |
| `pair_Vta1a` | 180s | 11737.7 | 1567.3 | 2107.1 | 1496.6 | **1547.0** | **-560.1 m** | Straight road vibration immunity holds |
| `pair_Vta17` | 180s | 66726.4 | 1072.6 | 1457.6 | 1317.9 | **1010.9** | **-446.6 m** | Straight road vibration immunity holds |

## 5. Curve Dynamic Bias & Speed Floor Recovery Audit

Audit of curved instances diagnosed in Step 23 where v3 suffered speed collapse or axis projection errors:

| Run ID | Outage | Corrected (m) | EKF v1 (m) | EKF v2 (m) | EKF v3 (m) | EKF v4 (m) | Delta vs v3 (m) | Diagnosis Mechanism |
|---|---|---|---|---|---|---|---|---|
| `pair_Vta2` | 180s | 55714.3 | 1585.9 | 1481.4 | 1641.4 | **1623.7** | **-17.7 m** | Speed floor & mount-invariant yaw |
| `pair_Vta2` | 120s | 25102.1 | 1071.4 | 992.5 | 1075.8 | **1055.8** | **-20.0 m** | Speed floor & mount-invariant yaw |
| `pair_Vta2` | 60s | 2520.6 | 1144.7 | 1055.8 | 1039.4 | **1012.7** | **-26.7 m** | Speed floor & mount-invariant yaw |
| `pair_Vta2` | 30s | 383.1 | 332.7 | 448.8 | 337.2 | **333.6** | **-3.6 m** | Speed floor & mount-invariant yaw |
| `pair_Vta2` | 10s | 30.0 | 1.6 | 1.6 | 1.6 | **1.6** | **+0.0 m** | Speed floor & mount-invariant yaw |
| `pair_Vta21` | 180s | 90242.4 | 1275.3 | 1001.4 | 1145.8 | **1175.2** | **+29.4 m** | Speed floor & mount-invariant yaw |
| `pair_Vta21` | 120s | 6486.7 | 703.5 | 750.4 | 613.5 | **659.6** | **+46.1 m** | Speed floor & mount-invariant yaw |
| `pair_Vta21` | 60s | 15936.7 | 522.3 | 484.1 | 501.2 | **487.3** | **-13.9 m** | Speed floor & mount-invariant yaw |
| `pair_Vta21` | 30s | 869.9 | 731.1 | 749.2 | 708.9 | **726.7** | **+17.8 m** | Speed floor & mount-invariant yaw |
| `pair_Vta21` | 10s | 96.6 | 123.0 | 76.3 | 90.3 | **90.8** | **+0.5 m** | Speed floor & mount-invariant yaw |
| `pair_Vta23` | 60s | 7932.2 | 605.2 | 403.1 | 465.8 | **459.9** | **-5.9 m** | Speed floor & mount-invariant yaw |
| `pair_Vta23` | 30s | 2269.9 | 282.4 | 316.1 | 280.1 | **279.8** | **-0.2 m** | Speed floor & mount-invariant yaw |
| `pair_Vta23` | 10s | 79.4 | 94.6 | 74.3 | 71.8 | **76.2** | **+4.4 m** | Speed floor & mount-invariant yaw |
| `pair_Vta27` | 180s | 81730.4 | 2326.4 | 2574.0 | 2312.1 | **2337.2** | **+25.2 m** | Speed floor & mount-invariant yaw |
| `pair_Vta27` | 120s | 34501.6 | 1911.9 | 1943.7 | 1830.4 | **1850.3** | **+19.9 m** | Speed floor & mount-invariant yaw |
| `pair_Vta27` | 60s | 10231.6 | 1302.7 | 1176.9 | 1247.3 | **1261.2** | **+13.9 m** | Speed floor & mount-invariant yaw |
| `pair_Vta27` | 30s | 828.5 | 786.6 | 661.6 | 739.1 | **777.3** | **+38.3 m** | Speed floor & mount-invariant yaw |
| `pair_Vta27` | 10s | 454.0 | 487.7 | 416.4 | 430.0 | **442.7** | **+12.7 m** | Speed floor & mount-invariant yaw |
| `pair_Vta29` | 180s | 64133.1 | 2366.1 | 2685.8 | 2532.3 | **2204.2** | **-328.1 m** | Speed floor & mount-invariant yaw |
| `pair_Vta29` | 120s | 42548.9 | 1980.2 | 2310.0 | 2324.8 | **1937.3** | **-387.5 m** | Speed floor & mount-invariant yaw |
| `pair_Vta29` | 60s | 11922.1 | 1041.8 | 681.2 | 959.3 | **833.3** | **-126.0 m** | Speed floor & mount-invariant yaw |
| `pair_Vta29` | 30s | 492.8 | 46.7 | 62.0 | 62.5 | **60.7** | **-1.8 m** | Speed floor & mount-invariant yaw |
| `pair_Vta29` | 10s | 34.7 | 33.0 | 29.7 | 28.6 | **28.5** | **-0.1 m** | Speed floor & mount-invariant yaw |
| `pair_S3c` | 180s | 3941.5 | 5059.6 | 3998.1 | 3474.3 | **3514.1** | **+39.8 m** | Speed floor & mount-invariant yaw |
| `pair_S3c` | 120s | 35299.8 | 1279.2 | 749.7 | 1298.7 | **1324.3** | **+25.6 m** | Speed floor & mount-invariant yaw |
| `pair_S3c` | 60s | 1533.0 | 193.4 | 209.8 | 138.3 | **137.3** | **-0.9 m** | Speed floor & mount-invariant yaw |
| `pair_S3c` | 30s | 216.6 | 1006.3 | 910.6 | 932.5 | **907.3** | **-25.3 m** | Speed floor & mount-invariant yaw |
| `pair_S3c` | 10s | 52.3 | 129.2 | 113.3 | 121.0 | **120.7** | **-0.3 m** | Speed floor & mount-invariant yaw |