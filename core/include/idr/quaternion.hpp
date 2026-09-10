#pragma once

#include "idr/vector3d.hpp"
#include <cmath>

namespace idr {

/**
 * @brief Unit quaternion representing 3D spatial rotation.
 *
 * Convention: q = [w, x, y, z] where w is scalar part.
 * Transforms a vector from body frame to navigation frame: v_nav = R(q) * v_body.
 */
struct Quaternion {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Quaternion() = default;
    constexpr Quaternion(double w_, double x_, double y_, double z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    /**
     * @brief Construct quaternion from rotation vector (angle * unit_axis).
     */
    static Quaternion fromRotationVector(const Vector3d& rot_vec) noexcept {
        const double theta = rot_vec.norm();
        if (theta < 1e-12) {
            // First-order Taylor expansion for near-zero rotation
            return Quaternion(1.0, 0.5 * rot_vec.x, 0.5 * rot_vec.y, 0.5 * rot_vec.z).normalized();
        }
        const double half_theta = 0.5 * theta;
        const double factor = std::sin(half_theta) / theta;
        return Quaternion(
            std::cos(half_theta),
            rot_vec.x * factor,
            rot_vec.y * factor,
            rot_vec.z * factor
        );
    }

    /**
     * @brief Construct initial horizontal attitude from heading (azimuth) in ENU navigation frame.
     *
     * In ENU (X=East, Y=North, Z=Up) and standard vehicle body frame (X=Right, Y=Forward, Z=Up),
     * a heading of psi (degrees clockwise from North) corresponds to rotating the body frame
     * such that Forward [0, 1, 0] points along [sin(psi), cos(psi), 0].
     */
    static Quaternion fromHeadingEnu(double heading_deg) noexcept {
        constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
        const double psi = heading_deg * kDegToRad;
        const double half_psi = 0.5 * psi;
        // Rotation about -Z_nav axis aligns Forward with azimuth psi
        return Quaternion(std::cos(half_psi), 0.0, 0.0, -std::sin(half_psi));
    }

    /**
     * @brief Construct attitude from heading and coarse leveling (accelerometer gravity direction).
     */
    static Quaternion fromHeadingAndLeveling(double heading_deg, const Vector3d& accel_body) noexcept {
        // First, orient horizontal heading
        Quaternion q_yaw = fromHeadingEnu(heading_deg);

        // Compute roll and pitch from specific force if gravity magnitude is reasonable
        const double norm = accel_body.norm();
        if (norm > 5.0 && norm < 15.0) {
            // Pitch (rotation about body X): atan2(f_y, sqrt(f_x^2 + f_z^2))
            const double pitch = std::atan2(accel_body.y, std::sqrt(accel_body.x * accel_body.x + accel_body.z * accel_body.z));
            // Roll (rotation about body Y): atan2(-f_x, f_z)
            const double roll = std::atan2(-accel_body.x, accel_body.z);

            const double hp = 0.5 * pitch;
            const double hr = 0.5 * roll;
            Quaternion q_pitch(std::cos(hp), std::sin(hp), 0.0, 0.0);
            Quaternion q_roll(std::cos(hr), 0.0, std::sin(hr), 0.0);

            return (q_yaw * q_pitch * q_roll).normalized();
        }

        return q_yaw;
    }

    /**
     * @brief Construct attitude from Euler angles in ENU (yaw clockwise from North, pitch, roll in radians).
     */
    static Quaternion fromEulerEnu(double yaw_deg, double pitch_rad, double roll_rad) noexcept {
        constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
        const double half_yaw = 0.5 * yaw_deg * kDegToRad;
        const Quaternion q_yaw(std::cos(half_yaw), 0.0, 0.0, -std::sin(half_yaw));

        const double half_pitch = 0.5 * pitch_rad;
        const Quaternion q_pitch(std::cos(half_pitch), std::sin(half_pitch), 0.0, 0.0);

        const double half_roll = 0.5 * roll_rad;
        const Quaternion q_roll(std::cos(half_roll), 0.0, std::sin(half_roll), 0.0);

        return (q_yaw * q_pitch * q_roll).normalized();
    }

    double normSq() const noexcept {
        return w * w + x * x + y * y + z * z;
    }

    double norm() const noexcept {
        return std::sqrt(normSq());
    }

    Quaternion normalized() const noexcept {
        const double n = norm();
        if (n > 1e-15) {
            return Quaternion(w / n, x / n, y / n, z / n);
        }
        return Quaternion(1.0, 0.0, 0.0, 0.0);
    }

    Quaternion conjugate() const noexcept {
        return Quaternion(w, -x, -y, -z);
    }

    /**
     * @brief Hamilton quaternion product: q_result = (*this) * q_other.
     */
    Quaternion operator*(const Quaternion& o) const noexcept {
        return Quaternion(
            w * o.w - x * o.x - y * o.y - z * o.z,
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w
        );
    }

    /**
     * @brief Rotate a 3D vector from body frame into navigation frame: v_nav = R(q) * v_body.
     */
    Vector3d rotate(const Vector3d& v) const noexcept {
        // Using Rodrigues-like formula: v + 2*q_v x (q_v x v + w*v)
        const Vector3d q_v(x, y, z);
        const Vector3d t = 2.0 * q_v.cross(v);
        return v + w * t + q_v.cross(t);
    }

    /**
     * @brief Extract vehicle heading in degrees clockwise from North in ENU frame.
     */
    double toHeadingEnu() const noexcept {
        // Vehicle Forward [0, 1, 0] rotated to ENU
        const Vector3d fwd_nav = rotate(Vector3d(0.0, 1.0, 0.0));
        constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
        // In ENU, X is East and Y is North. Azimuth = atan2(fwd_E, fwd_N)
        double heading = std::atan2(fwd_nav.x, fwd_nav.y) * kRadToDeg;
        if (heading < 0.0) {
            heading += 360.0;
        }
        return heading;
    }

    /**
     * @brief Propagate attitude forward given angular rate vector over dt seconds.
     */
    Quaternion propagate(const Vector3d& omega_body, double dt) const noexcept {
        const Vector3d rot_vec = omega_body * dt;
        const Quaternion delta_q = fromRotationVector(rot_vec);
        return ((*this) * delta_q).normalized();
    }
};

} // namespace idr
