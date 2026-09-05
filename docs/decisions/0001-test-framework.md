# ADR 0001: Choice of Unit Testing Framework for C++ Core Engine

## Status
Accepted

## Context
The core navigation engine (`core/`) requires an automated unit testing suite that can run cleanly across multiple environments:
- Local developer machines (Windows with CMake and Ninja/MinGW/Clang/MSVC)
- Continuous Integration runners (Linux GitHub Actions)
- Embedded Linux and edge toolchains

The prompt allows choosing between **Catch2** and **GoogleTest**.

## Decision
We choose **Catch2 (v3)** as the primary unit testing framework, integrated via CMake `FetchContent` / `find_package`, complemented by an offline test runner fallback.

## Rationale
1. **Modern C++ Idioms**: Catch2 is built natively for C++14/C++17, aligning cleanly with our C++17 core codebase.
2. **Readability and Expressiveness**: Catch2's `TEST_CASE` and `SECTION` structure allows natural hierarchical test organization, where test fixtures share setup code naturally without boilerplate classes.
3. **No External System Dependencies**: In CMake, Catch2 v3 can be automatically fetched via `FetchContent` in CI with shallow cloning, requiring zero manual package manager installation.
4. **Offline Resilience**: The test runner is structured with a preprocessor fallback (`#if __has_include(<catch2/catch_test_macros.hpp>)`) so developers in strictly offline environments can compile and run core smoke assertions without being blocked by network fetching.

## Consequences
- Clean test code with natural assertion syntax (`REQUIRE`, `CHECK`).
- Fast test builds with `ctest --test-dir build`.
- Zero manual host dependencies required for running unit tests.
