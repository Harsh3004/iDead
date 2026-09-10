#pragma once

#include "idr/imu_sample.hpp"
#include "idr/quaternion.hpp"
#include "idr/vector3d.hpp"

namespace idr {

/**
 * @brief Core Strapdown Inertial Navigation System (INS) mechanization in local ENU frame.
 *
 * Implements standard classical strapdown mechanics:
 * 1. Attitude Propagation: Quaternion integration of body angular rates.
 * 2. Specific Force Resolution: Rotate body accelerometer into ENU frame via attitude quaternion.
 * 3. Gravity Subtraction: a_nav = R_b^n * f_body - [0, 0, g0]^T.
 * 4. Velocity Integration: v_nav += a_nav * dt.
 * 5. Position Propagation: Local tangent-plane ENU integration and WGS-84 conversion.
 */
class StrapdownIns {
public:
    static constexpr double kGravity = 9.80665;        // Standard acceleration due to gravity (m/s^2)
    static constexpr double kEarthRadius = 6371000.0;  // Mean Earth radius in meters

    StrapdownIns() = default;
    ~StrapdownIns() = default;

    /**
     * @brief Initialize navigation state from coasting GNSS fix and initial accelerometer measurement.
     *
     * @param t0 Timestamp of initialization (seconds).
     * @param lat0 Initial WGS-84 latitude (degrees).
     * @param lon0 Initial WGS-84 longitude (degrees).
     * @param alt0 Initial altitude (meters).
     * @param speed_ms Initial scalar speed (m/s).
     * @param heading_deg Initial heading in degrees clockwise from North [0.0, 360.0).
     * @param initial_accel Initial body-frame accelerometer reading for coarse leveling.
     */
    void initialize(
        double t0,
        double lat0,
        double lon0,
        double alt0,
        double speed_ms,
        double heading_deg,
        const Vector3d& initial_accel = Vector3d(0.0, 0.0, kGravity)
    ) noexcept;

    /**
     * @brief Initialize navigation state with an explicit attitude quaternion and constant gyro bias.
     *
     * @param t0 Timestamp of initialization (seconds).
     * @param lat0 Initial WGS-84 latitude (degrees).
     * @param lon0 Initial WGS-84 longitude (degrees).
     * @param alt0 Initial altitude (meters).
     * @param speed_ms Initial scalar speed (m/s).
     * @param heading_deg Initial heading in degrees clockwise from North [0.0, 360.0).
     * @param q0 Initial attitude quaternion (incorporating leveling and heading).
     * @param gyro_bias Estimated constant gyro bias vector in body frame (rad/s).
     * @param initial_accel Initial body-frame accelerometer reading for gravity decoupling.
     */
    void initializeWithAttitude(
        double t0,
        double lat0,
        double lon0,
        double alt0,
        double speed_ms,
        double heading_deg,
        const Quaternion& q0,
        const Vector3d& gyro_bias = Vector3d(0.0, 0.0, 0.0),
        const Vector3d& initial_accel = Vector3d(0.0, 0.0, kGravity)
    ) noexcept;

    /**
     * @brief Integrate one incoming IMU sample forward in time.
     *
     * @param imu ImuSample containing timestamp, ax/ay/az (m/s^2), gx/gy/gz (rad/s).
     */
    void update(const ImuSample& imu) noexcept;

    /**
     * @brief Reset navigation state to uninitialized.
     */
    void reset() noexcept;

    // State accessors
    bool isInitialized() const noexcept { return initialized_; }
    double getTimestamp() const noexcept { return t_; }

    double getLatitude() const noexcept;
    double getLongitude() const noexcept;
    double getAltitude() const noexcept;

    double getSpeed() const noexcept;
    double getHeadingDeg() const noexcept;

    const Vector3d& getPositionEnu() const noexcept { return p_; }
    const Vector3d& getVelocityEnu() const noexcept { return v_; }
    const Quaternion& getAttitude() const noexcept { return q_; }
    const Vector3d& getAccelerationEnu() const noexcept { return last_accel_nav_; }
    const Vector3d& getGyroBias() const noexcept { return gyro_bias_; }

private:
    bool       initialized_{false};
    double     t_{0.0};
    Quaternion q_{1.0, 0.0, 0.0, 0.0};    // Body to ENU attitude quaternion
    Vector3d   v_{0.0, 0.0, 0.0};          // Velocity in ENU (East, North, Up) [m/s]
    Vector3d   p_{0.0, 0.0, 0.0};          // Position displacement in local ENU [m]
    Vector3d   last_accel_nav_{0.0, 0.0, 0.0}; // Kinematic acceleration in ENU [m/s^2]
    Vector3d   gyro_bias_{0.0, 0.0, 0.0};  // Constant body gyro bias to subtract (rad/s)

    double     lat0_{0.0};                 // Reference origin latitude (degrees)
    double     lon0_{0.0};                 // Reference origin longitude (degrees)
    double     alt0_{0.0};                 // Reference origin altitude (meters)
};

} // namespace idr
