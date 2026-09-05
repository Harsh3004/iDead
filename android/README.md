# Android Application Shell (`android/`)

This directory will contain the thin Android Kotlin shell that connects to the dependency-free C++17 core engine via JNI (Java Native Interface).

## Architectural Role

- Collects phone IMU hardware sensor events (`Sensor.TYPE_ACCELEROMETER`, `Sensor.TYPE_GYROSCOPE`, `Sensor.TYPE_MAGNETIC_FIELD`) and Android `Location` fixes.
- Maps raw sensor events into `ImuSample` and `GnssFix` structs.
- Invokes the compiled C++ shared library (`libidr_core.so`) in real time.
- Renders live dead-reckoning trajectory and GNSS outage notifications in a lightweight UI.

> **Status**: Empty scaffold placeholder. Full implementation scheduled for Phase 3.
