# Phone-Based Inertial Dead-Reckoning (IDR) Navigation Stack

A phone-based inertial dead-reckoning navigation engine developed for the ISRO hackathon. The system features a single dependency-free C++17 core engine consuming abstract IMU and GNSS streams, shared across three frontends: an offline CSV replay evaluation harness, an Android mobile application via JNI, and a Linux edge daemon for ~200 Hz FOG-grade IMUs. A Python data and evaluation pipeline handles dataset ingestion (IO-VNBD), synthetic GNSS outage simulation, model training, and benchmark leaderboard scoring.

## Repository Layout

```
idr-project/
├── core/                     # C++17 engine, zero external dependencies
│   ├── include/idr/          # Public headers (imu_sample.hpp, gnss_fix.hpp, engine.hpp)
│   ├── src/                  # Implementation (engine.cpp)
│   ├── tests/                # Unit tests (Catch2)
│   └── CMakeLists.txt        # Core build specification
├── dataeval/                 # Python data & evaluation pipeline
│   ├── ingest/               # Ingestion parsers for IO-VNBD dataset
│   ├── harness/              # Outage simulation, replay harness, and metric scoring
│   ├── training/             # Model training routines (ZUPT, orientation)
│   ├── pyproject.toml        # Package definition (numpy, pandas, pyarrow)
│   └── README.md
├── android/                  # Android app shell (Phase 3 Kotlin/JNI scaffold)
│   └── README.md
├── edge/                     # Linux edge CLI/daemon shell (Phase 3 scaffold)
│   └── README.md
├── data/                     # Gitignored data storage (raw + processed datasets)
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── docs/                     # Architectural specifications and ADRs
│   ├── architecture.md       # Architecture reference document
│   └── decisions/            # Architectural Decision Records (ADRs)
│       └── 0001-test-framework.md
├── results/                  # Benchmark tracking and visual outputs
│   ├── leaderboard.csv       # Standardized metric leaderboard (header schema)
│   └── plots/.gitkeep
├── .github/workflows/
│   └── ci.yml                # CI workflow (CMake build, C++ test, Python smoke test)
├── CMakeLists.txt            # Top-level CMake entry point
├── .gitignore
└── README.md
```

## How to Build and Test

### 1. C++17 Core Engine (`core/`)

The core library requires CMake (>= 3.16) and any standard C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+).

#### Build from Repository Root:
```bash
# Configure build directory
cmake -B build

# Compile library and test executable
cmake --build build

# Run unit tests via CTest
ctest --test-dir build --output-on-failure
```

#### Standalone Build in `core/`:
```bash
cd core
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

---

### 2. Python Data/Evaluation Pipeline (`dataeval/`)

Requires Python 3.9+.

#### Installation (Editable Mode):
```bash
# From repository root
pip install -e ./dataeval

# Or from inside dataeval/
cd dataeval
pip install -e .
```

#### Run Verification Smoke Test:
```bash
python -c "import dataeval.ingest, dataeval.harness, dataeval.training; print('All dataeval packages import cleanly')"
```

---

### 3. Continuous Integration

Automated builds and tests run on every pull request and push to main via GitHub Actions (`.github/workflows/ci.yml`), verifying:
- Clean C++17 build with CMake & Ninja on Ubuntu.
- Passing unit test execution via CTest.
- Successful editable installation and package import smoke testing of `dataeval`.
