# ADR 0006: Lightweight CI Fixture Subset Strategy

## Status
Accepted

## Context
The project encompasses an end-to-end data processing, synchronization, evaluation, and C++ dead-reckoning pipeline across 39 hours of driving data (564 raw CSVs, 1.71 GB IO-VNBD dataset, 72 synchronized pairs, 377 outage instances, and 754 leaderboard rows).

Running the entire data ingestion, sync, outage simulation, replay, and scoring pipeline from raw data on every commit in Continuous Integration (CI) is impractical:
1. **Network & Storage Constraints**: Hosted CI runners (e.g. GitHub Actions) have checkout size and cache limits. Uploading 1.71 GB of raw dataset files to the git repository would cause repository bloat and slow down every `git clone`.
2. **CI Execution Time**: Re-running all 377 outages and 754-row evaluations takes 10–15 minutes, which hinders rapid development feedback.
3. **Primary Purpose of CI**: The role of CI is to verify that code builds cleanly, interfaces remain compatible, unit logic passes, and pipeline stages run regression-free on real-shaped inputs—not to re-benchmarking the entire 39-hour dataset.

## Decision
We adopt a **Lightweight Checked-In Fixture Subset** strategy:
1. **Location**: Fixture files are stored in `dataeval/tests/fixtures/`, version-controlled directly in git. This path is outside `data/` and is therefore unaffected by `.gitignore` rules that ignore `data/raw/*` and `data/processed/*`.
2. **Scope & Footprint**:
   - We extract a minimal, multi-stage slice of representative runs (e.g. `Vfa01` paired CAN/phone drive).
   - Slices cover all pipeline stages:
     - `raw/`: 30-row raw CAN CSV (`V-Vfa01_slice.csv`) and 30-row raw Smartphone CSV (`S-Vfa01_slice.csv`).
     - `processed/vehicle_can/`: 30-row canonical CAN Parquet.
     - `processed/smartphone/`: 30-row canonical Smartphone Parquet.
     - `processed/paired/`: 50-row synchronized 10 Hz Parquet (`pair_Vfa01.parquet`) and pairing manifest.
     - `processed/outages/`: 50-row outage-masked Parquet and outage manifest.
     - `processed/cpp_predictions/`: 50-row C++ strapdown prediction CSV.
     - `results/`: Mini 2-row leaderboard CSV and summary CSV.
   - Total disk footprint is $< 100\text{ KB}$, enabling instantaneous checkouts in CI.
3. **CI Execution Scope**:
   - CI builds the C++17 core engine and runs CTest (`idr_core_tests`, `idr_strapdown_tests`).
   - CI installs `dataeval` and runs the complete unit test suite (`python -m unittest discover -s dataeval/tests -p "test_*.py"`).
   - CI runs an integration test suite (`test_ci_pipeline.py`) exercising each pipeline stage end-to-end on the fixture subset.
   - Full dataset regeneration, 754-row leaderboard re-scoring, and production plot generation remain offline/local developer workflows per Steps 8–11.

## Consequences
- **Instant CI Runs**: CI jobs complete in under 2 minutes with zero external dataset downloads.
- **High Test Fidelity**: Because fixtures are derived directly from actual vehicle CAN and smartphone hardware logs rather than mock synthetic arrays, tests catch real-world parsing, unit conversion, and schema regression bugs.
- **Repository Cleanliness**: The repository remains small and lightweight while guaranteeing full reproducibility outside the local machine.
