#pragma once

#include "idr/imu_sample.hpp"
#include "idr/gnss_fix.hpp"

namespace idr {

/**
 * @brief Core Inertial Dead-Reckoning Engine.
 *
 * Consumes abstract ImuSample and optional GnssFix streams.
 * In this initial stage, stores the latest sample to validate the
 * build system and cross-platform frontend interface.
 */
class IdrEngine {
public:
    IdrEngine();
    ~IdrEngine() = default;

    // Non-copyable, movable
    IdrEngine(const IdrEngine&) = delete;
    IdrEngine& operator=(const IdrEngine&) = delete;
    IdrEngine(IdrEngine&&) noexcept = default;
    IdrEngine& operator=(IdrEngine&&) noexcept = default;

    /**
     * @brief Process an incoming IMU sample (body frame).
     * @param sample Monotonic timestamp, accel (m/s^2), gyro (rad/s), mag (uT).
     */
    void processImu(const ImuSample& sample);

    /**
     * @brief Process an incoming GNSS position fix.
     * @param fix Timestamp, WGS-84 coordinates, speed, accuracy metrics.
     */
    void processGnss(const GnssFix& fix);

    /**
     * @brief Retrieve the most recently received IMU sample.
     */
    const ImuSample& getLastImu() const;

    /**
     * @brief Retrieve the most recently received GNSS fix.
     */
    const GnssFix& getLastGnss() const;

    /**
     * @brief Check whether at least one IMU sample has been processed.
     */
    bool hasImu() const noexcept;

    /**
     * @brief Check whether at least one GNSS fix has been processed.
     */
    bool hasGnss() const noexcept;

    /**
     * @brief Reset internal states and stored samples.
     */
    void reset() noexcept;

private:
    ImuSample last_imu_{};
    GnssFix   last_gnss_{};
    bool      has_imu_{false};
    bool      has_gnss_{false};
};

} // namespace idr
