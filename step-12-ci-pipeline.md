# Step 12 — CI Pipeline: Full Harness on Every Commit

## Context (what's real on disk right now — read before starting)

- Two independent test suites currently pass locally, and only locally:
  - **C++:** 2 CTest suites (`idr_core_tests`, `idr_strapdown_tests`), built via
    CMake + Ninja on a 32-bit MinGW GCC 6.3.0 Windows toolchain (per ADR 0001,
    ADR 0005 — Arrow C++ was ruled out on this toolchain, hence the CSV
    replay-cache I/O approach).
  - **Python:** 58 unit tests (`dataeval/tests/`), run via whatever runner
    Steps 1–11 have used so far (check the actual invocation in use — e.g.
    `pytest` or `python -m unittest discover` — before assuming; don't
    introduce a second test runner).
- Full pipeline reproducibility already exists end-to-end locally: Parquet
  conversion (Steps 3–4) → sync (Step 5) → split (Step 6) → outages (Step 7)
  → scoring harness + baseline (Step 8) → C++ strapdown replay (Step 9) →
  scoring C++ predictions (Step 10) → plots (Step 11). This step does **not**
  re-run that full data pipeline on every commit — the raw datasets
  (1.71 GB, IO-VNBD) are not meant to live in CI and regenerating everything
  from scratch is not this step's job.
- No commit/push history has been confirmed to me since Step 1 (which noted
  "no commits/pushes yet"). **Open decision — confirm before/while
  implementing, not after:**
  1. Is this repo already pushed to a Git host (GitHub, GitLab, etc.)? If
     not, this step needs a `git init` + first push before any CI config can
     run, and you'll need to tell me which host/plan you're using.
  2. Given the 32-bit MinGW toolchain is specific to your local machine, can
     hosted CI runners (e.g. GitHub Actions' `windows-latest`, which is
     64-bit MSVC/MinGW) build the C++ core as-is, or does the CMake config
     need a toolchain-agnostic path? Do **not** silently swap the toolchain
     without reporting the change — if CI needs a different compiler than
     local dev, that's a real deviation to flag back to me.
  3. Because the 90+97 raw source Parquet files and 378 outage/prediction
     files are sizeable and derived, decide (and report) whether CI
     regenerates a **small fixture subset** from checked-in raw fixtures, or
     whether processed Parquet/outage/prediction artifacts are checked into
     the repo (or an LFS/artifact store) for CI to consume directly. Prefer
     a small fixture subset (e.g. 2–3 runs' worth) checked into the repo
     specifically for CI, rather than committing the full 1.71 GB dataset or
     all 754 leaderboard rows' worth of intermediate files — this keeps CI
     fast and the repo small. This is a new decision, not one made in Steps
     1–11; ADR it (e.g. `docs/decisions/0006-ci-fixture-strategy.md`).

## Objective

Wire the existing C++ CTest suite and Python test suite into an automated CI
pipeline that runs on every commit/push, using a small, fast, checked-in
fixture subset — not the full 1.71 GB dataset — so Phase 0's exit gate
(leaderboard + plots checked into the repo, verified by a green pipeline) is
closed.

## In scope

### 1. Fixture subset for CI
- Select 2–3 small representative runs (prefer ones already used in Step 11's
  spot-checks/trajectory plots, e.g. `Vfa01`, `Y1`) and derive a minimal
  fixture set covering: raw CSV → Parquet → paired → outage → prediction, at
  each pipeline stage the tests actually exercise. Store under something
  like `dataeval/tests/fixtures/` or `data/fixtures/` — your call, follow
  whatever fixture convention Steps 1–11's existing tests already use if one
  exists; don't invent a second one.
- Do **not** commit the full `data/raw/`, `data/processed/`, or
  `results/leaderboard.csv` (754 rows) to satisfy this — CI should prove the
  *code* works on real-shaped data, not replay the full 39-hour dataset.

### 2. CI config
- One pipeline definition (e.g. `.github/workflows/ci.yml` if GitHub, or the
  equivalent for whatever host is confirmed in the open decision above),
  triggered on every push and pull request, with (at minimum) two jobs/steps:
  1. **C++ build + test:** configure/build via CMake + Ninja, run both
     CTest suites, fail the pipeline on any test failure or build error.
  2. **Python test:** install `dataeval` (`pip install -e .`), run the full
     58-test suite against the fixture subset, fail on any failure.
- Both jobs must run from a clean checkout — no dependency on local machine
  state, no absolute Windows paths (`d:/Code/SIH26/...`) baked into scripts;
  use repo-relative paths throughout, since CI runners won't have `d:/`.
- Cache dependencies (pip packages, CMake build dir) between runs where the
  CI platform supports it, to keep run time reasonable — but this is a
  nice-to-have, not a blocker if it adds complexity.

### 3. Status visibility
- Add a CI status badge to the top-level `README.md` (create one if it
  doesn't exist yet) so pass/fail is visible without opening the pipeline.

### 4. Documentation
- ADR for the fixture strategy decision (point 3 above).
- Update `docs/` or the README with: how to run the same checks locally
  before pushing (so CI failures aren't a surprise), and what's
  intentionally excluded from CI (full dataset regeneration, 754-row
  leaderboard reproduction, plot generation — those remain manual/local
  steps per Steps 8–11).

## Out of scope
- No re-running the full 378-instance scoring/replay pipeline in CI — too
  slow and not this step's job; fixture-scale coverage is sufficient to
  catch regressions.
- No deployment, packaging, or release automation (that's Phase 3+).
- No linting/formatting enforcement unless it's trivial to bolt on — don't
  let that scope-creep this step; a green test pipeline is the bar.
- No changes to the actual scoring/strapdown/plotting logic — this step only
  wires existing, working code into CI.

## Acceptance criteria
- [ ] Git host and toolchain-compatibility questions above answered and
      reported back (not just implemented silently).
- [ ] Fixture subset checked into the repo, small enough to review at a
      glance (report its size).
- [ ] CI pipeline definition committed, triggers on push/PR.
- [ ] Both C++ CTest suites and all 58 Python tests pass in CI using only
      the fixture subset — report the actual CI run result (green/red) with
      a link or log excerpt, not just "should work."
- [ ] No absolute local paths (`d:/Code/SIH26/...`) remain in any code path
      CI exercises.
- [ ] README has a CI status badge.
- [ ] ADR written for the fixture strategy.
- [ ] Report back: which host/runner was used, whether the toolchain matched
      local dev or had to change, fixture subset contents/size, and the
      actual pass/fail result of the first real CI run.

## Handoff

This is the last step needed to close Phase 0's exit gate: leaderboard CSV
with 2 baseline configs (done, Steps 8+10), plots (done, Step 11), and now a
green CI pipeline proving it's all reproducible outside your local machine.
Implement this, get a real CI run to go green, then report back with the
answers to the three open decisions above plus the actual run result. Once
that's confirmed, I'll mark Phase 0 closed and we'll move to Phase 1
(Step 14 — Module A static-phase gravity/pitch/roll + gyro bias), which is
your first ML/estimation step.
