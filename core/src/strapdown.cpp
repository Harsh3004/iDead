#include "idr/strapdown.hpp"
#include <cmath>

namespace idr {

void StrapdownIns::initialize(
    double t0,
    double lat0,
    double lon0,
    double alt0,
    double speed_ms,
    double heading_deg,
    const Vector3d& initial_accel
) noexcept {
    initializeWithAttitude(
        t0, lat0, lon0, alt0, speed_ms, heading_deg,
        Quaternion::fromHeadingEnu(heading_deg),
        Vector3d(0.0, 0.0, 0.0),
        initial_accel
    );
}

void StrapdownIns::initializeWithAttitude(
    double t0,
    double lat0,
    double lon0,
    double alt0,
    double speed_ms,
    double heading_deg,
    const Quaternion& q0,
    const Vector3d& gyro_bias,
    const Vector3d& initial_accel
) noexcept {
    t_ = t0;
    lat0_ = lat0;
    lon0_ = lon0;
    alt0_ = alt0;

    // Use explicit attitude quaternion (already normalized)
    q_ = q0.normalized();
    gyro_bias_ = gyro_bias;

    // Initial velocity along initial heading in ENU
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double psi_rad = heading_deg * kDegToRad;
    v_ = Vector3d(
        speed_ms * std::sin(psi_rad),
        speed_ms * std::cos(psi_rad),
        0.0
    );

    // Initial position in local ENU is origin (0, 0, 0)
    p_ = Vector3d(0.0, 0.0, 0.0);

    // Initial kinematic acceleration with gravity decoupling
    const Vector3d gravity_nav(0.0, 0.0, -kGravity);
    last_accel_nav_ = q_.rotate(initial_accel) + gravity_nav;

    initialized_ = true;
}

void StrapdownIns::update(const ImuSample& imu) noexcept {
    if (!initialized_) {
        return;
    }

    const double dt = imu.t - t_;
    if (dt <= 0.0 || dt > 1.0) {
        // Ignore negative or excessive time gaps
        t_ = imu.t;
        return;
    }

    // 1. Attitude Propagation via quaternion rotation vector (subtracting gyro bias)
    const Vector3d omega_body(
        imu.gx - gyro_bias_.x,
        imu.gy - gyro_bias_.y,
        imu.gz - gyro_bias_.z
    );
    q_ = q_.propagate(omega_body, dt);

    // 2. Specific Force Resolution into ENU Navigation Frame
    const Vector3d f_body(imu.ax, imu.ay, imu.az);
    const Vector3d f_nav = q_.rotate(f_body);

    // 3. Gravity Subtraction
    const Vector3d gravity_nav(0.0, 0.0, -kGravity);
    const Vector3d accel_nav = f_nav + gravity_nav;

    // 4. Trapezoidal Integration for Velocity
    const Vector3d delta_v = (last_accel_nav_ + accel_nav) * (0.5 * dt);
    const Vector3d v_prev = v_;
    v_ = v_ + delta_v;

    // 5. Trapezoidal Integration for Position in Local ENU
    const Vector3d delta_p = (v_prev + v_) * (0.5 * dt);
    p_ = p_ + delta_p;

    // Update state for next step
    last_accel_nav_ = accel_nav;
    t_ = imu.t;
}

void StrapdownIns::reset() noexcept {
    initialized_ = false;
    t_ = 0.0;
    q_ = Quaternion(1.0, 0.0, 0.0, 0.0);
    v_ = Vector3d(0.0, 0.0, 0.0);
    p_ = Vector3d(0.0, 0.0, 0.0);
    last_accel_nav_ = Vector3d(0.0, 0.0, 0.0);
    gyro_bias_ = Vector3d(0.0, 0.0, 0.0);
    lat0_ = 0.0;
    lon0_ = 0.0;
    alt0_ = 0.0;
}

double StrapdownIns::getLatitude() const noexcept {
    constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
    // delta_lat = delta_North / R
    const double delta_lat_deg = (p_.y / kEarthRadius) * kRadToDeg;
    return lat0_ + delta_lat_deg;
}

double StrapdownIns::getLongitude() const noexcept {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
    const double lat_rad = lat0_ * kDegToRad;
    const double cos_lat = std::cos(lat_rad);
    if (std::abs(cos_lat) < 1e-6) {
        return lon0_;
    }
    // delta_lon = delta_East / (R * cos(lat))
    const double delta_lon_deg = (p_.x / (kEarthRadius * cos_lat)) * kRadToDeg;
    double lon = lon0_ + delta_lon_deg;

    // Normalize to [-180, 180)
    while (lon >= 180.0) lon -= 360.0;
    while (lon < -180.0) lon += 360.0;
    return lon;
}

double StrapdownIns::getAltitude() const noexcept {
    return alt0_ + p_.z;
}

double StrapdownIns::getSpeed() const noexcept {
    return std::sqrt(v_.x * v_.x + v_.y * v_.y);
}

double StrapdownIns::getHeadingDeg() const noexcept {
    return q_.toHeadingEnu();
}

} // namespace idr
