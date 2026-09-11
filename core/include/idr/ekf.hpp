#pragma once

#include "idr/vector3d.hpp"
#include "idr/quaternion.hpp"
#include "idr/matrix.hpp"
#include "idr/imu_sample.hpp"
#include <vector>

namespace idr {

/**
 * @brief Tuning parameters and sensor noise specifications for 15-state ES-EKF.
 *
 * Direct port of validated Python prototype (dataeval/estimation/ekf.py:EkfConfig).
 */
struct EkfConfig {
    double sigma_accel_noise{0.10};       // [m/s^2]
    double sigma_gyro_noise{0.01};        // [rad/s]
    double sigma_accel_bias{0.001};       // [m/s^2/sqrt(s)]
    double sigma_gyro_bias{0.0001};       // [rad/s/sqrt(s)]

    double sigma_zupt{0.05};              // [m/s]
    double sigma_nhc_lat{0.15};           // [m/s]
    double sigma_nhc_vert{0.15};          // [m/s]

    // ZUPT detector thresholds
    double zupt_gyro_thresh_rad_s{0.05};     // ~2.8 deg/s
    double zupt_accel_std_thresh_m_s2{0.25}; // vibration threshold
    double zupt_accel_mag_window_s{0.6};     // window duration for stillness check
    double zupt_speed_gate_m_s{2.5};         // speed gate to avoid false triggers

    // Step 20, Step 22 & Step 24: Short-horizon settling and curvature-adaptive NHC
    double nhc_settle_duration_s{2.0};        // Settling time constant tau [s]
    double nhc_settle_sigma_extra{4.0};       // Initial extra sigma at t=t0 [m/s]
    double nhc_curv_c_coeff{2.0};             // Kinematic centripetal accel coeff (k_c) for a_c = v * w_yaw
    double nhc_curv_lat_coeff{0.0};           // Deprecated Step 20 raw accel coeff
    double nhc_curv_yaw_coeff{0.0};           // Curvature inflation coefficient for direct yaw rate
    double nhc_curv_lpf_cutoff_hz{2.0};       // Step 24: Low-pass filter cutoff for yaw rate [Hz]
    bool nhc_curv_use_speed_floor{true};      // Step 24: Pre-outage speed floor v_curv = max(v, v0)
    bool nhc_curv_use_nav_yaw{true};          // Step 24: Mount-orientation-invariant nav-frame yaw rate
};

struct ImuBufferedSample {
    double t{0.0};
    Vector3d f_body;
    Vector3d omega_body;
};

/**
 * @brief 15-State Error-State Extended Kalman Filter (ES-EKF).
 *
 * State vector: delta_x = [delta_p (3), delta_v (3), delta_theta (3), b_gyro (3), b_accel (3)] in R^15.
 * Coupled with:
 * - Nominal state propagation (quaternion strapdown integration)
 * - Zero-Velocity Updates (ZUPT) during stationary intervals
 * - Non-Holonomic Constraints (NHC) during forward vehicle motion
 * - Joseph stabilized covariance updates ensuring positive definiteness
 */
class ErrorStateEkf {
public:
    static constexpr double kGravity = 9.80665;
    static constexpr double kEarthRadius = 6371000.0;
    static constexpr double kPi = 3.14159265358979323846;

    explicit ErrorStateEkf(const EkfConfig& config = EkfConfig());

    void initialize(
        double t0,
        double lat0,
        double lon0,
        double alt0,
        double speed_ms,
        double heading_deg,
        const Vector3d& initial_accel = Vector3d(0.0, 0.0, kGravity)
    );

    void initializeWithAttitude(
        double t0,
        double lat0,
        double lon0,
        double alt0,
        double speed_ms,
        double heading_deg,
        const Quaternion& q0,
        const Vector3d& b_gyro0 = Vector3d(0.0, 0.0, 0.0),
        const Vector3d& b_accel0 = Vector3d(0.0, 0.0, 0.0),
        const Vector3d& initial_accel = Vector3d(0.0, 0.0, kGravity)
    );

    void predict(double t, const Vector3d& f_body, const Vector3d& omega_body);

    bool isStationary() const;

    bool updateZupt();

    bool updateNhc();

    /**
     * @brief High-level update pipeline: predict + (isStationary ? updateZupt : (speed > 0.5 ? updateNhc : none)).
     */
    bool update(const ImuSample& sample);

    // Geodetic outputs
    double getLatitude() const noexcept;
    double getLongitude() const noexcept;
    double getAltitude() const noexcept;
    double getSpeed() const noexcept;
    double getHeadingDeg() const noexcept;

    // Telemetry & States
    Vector3d getPositionEnu() const noexcept { return p_enu_; }
    Vector3d getVelocityEnu() const noexcept { return v_enu_; }
    Quaternion getAttitude() const noexcept { return q_; }
    Vector3d getGyroBias() const noexcept { return b_gyro_; }
    Vector3d getAccelBias() const noexcept { return b_accel_; }
    const Matrix<15, 15>& getCovariance() const noexcept { return P_; }

    double getCovarianceTrace() const noexcept { return P_.trace(); }
    double getPositionUncertainty() const noexcept {
        return std::sqrt(P_(0, 0) + P_(1, 1));
    }

    bool isInitialized() const noexcept { return is_initialized_; }
    double getTimestamp() const noexcept { return t_; }
    const EkfConfig& getConfig() const noexcept { return config_; }

    // Diagnostic & testing getters for adaptive NHC measurement noise
    double computeNhcSigmaLat() const noexcept;
    double computeNhcSigmaVert() const noexcept;
    double getSpeedFloor() const noexcept { return v0_; }
    double getOmegaYawFilt() const noexcept { return std::abs(omega_yaw_filt_); }

private:
    void initCovariance();

    EkfConfig config_;
    bool is_initialized_{false};

    // Nominal state
    double t_{0.0};
    double t0_{0.0};
    Vector3d p_enu_{0.0, 0.0, 0.0};
    Vector3d v_enu_{0.0, 0.0, 0.0};
    Quaternion q_{1.0, 0.0, 0.0, 0.0};
    Vector3d b_gyro_{0.0, 0.0, 0.0};
    Vector3d b_accel_{0.0, 0.0, 0.0};

    // Geodetic origin
    double lat0_{0.0};
    double lon0_{0.0};
    double alt0_{0.0};

    // Step 24: Pre-outage speed floor and low-pass filtered horizontal yaw rate
    double v0_{0.0};
    double omega_yaw_filt_{0.0};
    double last_t_lpf_{-1.0};

    // Last IMU readings for trapezoidal integration & adaptive noise calculation
    Vector3d last_accel_nav_{0.0, 0.0, 0.0};
    Vector3d last_f_body_{0.0, 0.0, 0.0};
    Vector3d last_omega_body_{0.0, 0.0, 0.0};

    // Error covariance P in R^{15x15}
    Matrix<15, 15> P_;

    // Buffer for stillness detection
    std::vector<ImuBufferedSample> imu_buffer_;
};

} // namespace idr
