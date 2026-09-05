# Data & Evaluation Pipeline (`dataeval/`)

This package contains the Python tools for data ingestion, dataset parsing, synthetic GNSS outage simulation, offline trajectory replay harness, metric computation, and ML model training for the Phone-Based Inertial Dead-Reckoning (IDR) navigation stack.

## Subpackage Layout

- `dataeval/ingest/`: Parsers for the IO-VNBD dataset, raw sensor CSV cleaning, time synchronization, and coordinate conversion into standardized parquet/dataframe structures.
- `dataeval/harness/`: GNSS outage injector (e.g., simulating 10s, 30s, 60s GNSS loss), trajectory replay driver feeding the C++ engine, and evaluation metric scoring against ground truth.
- `dataeval/training/`: Training routines, feature extraction, neural network architectures (e.g., learned ZUPT, step/stride estimators, heading drift compensators), and model export (ONNX).

## Installation

We use standard Python packaging (`pyproject.toml`) with `pip` (or optionally `uv`).

### Using `pip` (Editable Mode)

From this directory:
```bash
pip install -e .
```
Or from the project root:
```bash
pip install -e ./dataeval
```

### Using `uv` (Optional)

```bash
uv pip install -e .
```

## Core Dependencies

At this stage, `dataeval` intentionally relies only on lightweight core data structures:
- `numpy`: Numerical operations and vector calculations
- `pandas`: Tabular time-series processing
- `pyarrow`: High-performance columnar serialization for dataset caching

Machine learning libraries (`torch`, `onnx`, etc.) and geospatial libraries (`osmium`, `shapely`, `geopy`) will be introduced in subsequent steps as specific modules are built.

## Verification

To verify that the packages are importable after installation:
```bash
python -c "import dataeval.ingest, dataeval.harness, dataeval.training; print('dataeval pipeline successfully initialized')"
```
