# Linux Edge CLI & Daemon Shell (`edge/`)

This directory will contain the thin Linux binary CLI and daemon shell wrapping the same dependency-free C++17 core engine.

## Architectural Role

- Interfaces with high-frequency (~200 Hz) FOG-grade (Fiber Optic Gyroscope) IMUs over serial / CAN / SPI.
- Consumes real-time IMU packets and optional GNSS NMEA/UBX streams.
- Runs continuous dead-reckoning with minimal latency on embedded edge hardware (e.g., Raspberry Pi, Jetson, or industrial x86/ARM Linux boards).

> **Status**: Empty scaffold placeholder. Full implementation scheduled for Phase 3.
