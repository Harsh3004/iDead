# IO-VNBD Train / Validation / Test Split Summary

## Partition Ratios

| Split | Groups | Drives | Duration (hrs) | Duration % | Distance (km) | Distance % | Paired Runs | Paired Duration % of Split |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 29 | 83 | 47.98 | 68.0% | 2784.3 | 68.5% | 48 | 32.5% |
| **VAL** | 4 | 23 | 11.26 | 15.9% | 687.1 | 16.9% | 21 | 88.0% |
| **TEST** | 8 | 9 | 11.37 | 16.1% | 591.4 | 14.6% | 3 | 36.4% |
| **TOTAL** | 41 | 115 | 70.61 | 100.0% | 4062.8 | 100.0% | 72 | 42.0% |

## Hard-Cohort Representation in Evaluation Splits

| Cohort / Flag | Present in Train | Present in Val | Present in Test | Representative Evaluated Groups |
|:---|:---:|:---:|:---:|:---|
| `phone:low_sample_rate` | Yes | Yes | Yes | Val: Driver_H_A12, Test: Driver_H_A2, Test: Driver_H_A9 |
| `phone:burst_sampling` | Yes | Yes | Yes | Val: Driver_F_T6, Test: Driver_F_T5 |
| `phone:no_magnetometer` | Yes | Yes | Yes | Val: Driver_F_T6, Test: Driver_F_T5 |
| `stationary_run` | No | Yes | No | Val: Driver_E_Vw |
| `phone:clock_reset` | Yes | Yes | Yes | Val: Driver_A_S2, Test: Driver_D_Y1 |
| `low_confidence_offset` | Yes | Yes | Yes | Val: Driver_E_Vw, Test: Driver_D_Y1 |