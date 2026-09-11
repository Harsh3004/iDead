#include "idr/ekf.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace idr {

ErrorStateEkf::ErrorStateEkf(const EkfConfig& config)
    : config_(config) {
    initCovariance();
}

void ErrorStateEkf::initCovariance() {
    P_ = Matrix<15, 15>::zeros();
    // Position error covariance: 1.0 m^2 horizontal, 4.0 m^2 vertical
    P_(0, 0) = 1.0;  P_(1, 1) = 1.0;  P_(2, 2) = 4.0;
    // Velocity error covariance: 0.25 (m/s)^2
    P_(3, 3) = 0.25; P_(4, 4) = 0.25; P_(5, 5) = 0.25;
    // Attitude error covariance: 0.001 rad^2 horizontal, 0.002 rad^2 vertical
    P_(6, 6) = 0.001; P_(7, 7) = 0.001; P_(8, 8) = 0.002;
    // Gyro bias covariance: 1.0e-4 (rad/s)^2
    P_(9, 9) = 1.0e-4; P_(10, 10) = 1.0e-4; P_(11, 11) = 1.0e-4;
    // Accel bias covariance: 0.01 (m/s^2)^2
    P_(12, 12) = 0.01; P_(13, 13) = 0.01; P_(14, 14) = 0.01;
}

void ErrorStateEkf::initialize(
    double t0,
    double lat0,
    double lon0,
    double alt0,
    double speed_ms,
    double heading_deg,
    const Vector3d& initial_accel
) {
    const double psi_rad = heading_deg * (kPi / 180.0);
    const double half_psi = 0.5 * psi_rad;
    const Quaternion q_flat = Quaternion(std::cos(half_psi), 0.0, 0.0, -std::sin(half_psi)).normalized();

    initializeWithAttitude(
        t0, lat0, lon0, alt0, speed_ms, heading_deg,
        q_flat,
        Vector3d(0.0, 0.0, 0.0),
        Vector3d(0.0, 0.0, 0.0),
        initial_accel
    );
}

void ErrorStateEkf::initializeWithAttitude(
    double t0,
    double lat0,
    double lon0,
    double alt0,
    double speed_ms,
    double heading_deg,
    const Quaternion& q0,
    const Vector3d& b_gyro0,
    const Vector3d& b_accel0,
    const Vector3d& initial_accel
) {
    t_ = t0;
    t0_ = t0;
    lat0_ = lat0;
    lon0_ = lon0;
    alt0_ = alt0;

    p_enu_ = Vector3d(0.0, 0.0, 0.0);

    const double psi_rad = heading_deg * (kPi / 180.0);
    v_enu_ = Vector3d(speed_ms * std::sin(psi_rad), speed_ms * std::cos(psi_rad), 0.0);

    q_ = (q0.normSq() > 1e-6) ? q0.normalized() : Quaternion(1.0, 0.0, 0.0, 0.0);
    b_gyro_ = b_gyro0;
    b_accel_ = b_accel0;

    const Vector3d gravity_nav(0.0, 0.0, -kGravity);
    last_accel_nav_ = q_.rotate(initial_accel - b_accel_) + gravity_nav;
    last_f_body_ = initial_accel;
    last_omega_body_ = Vector3d(0.0, 0.0, 0.0);

    // Step 24: Pre-outage speed floor and low-pass filtered horizontal yaw rate
    v0_ = speed_ms;
    omega_yaw_filt_ = 0.0;
    last_t_lpf_ = -1.0;

    initCovariance();
    imu_buffer_.clear();
    is_initialized_ = true;
}

void ErrorStateEkf::predict(double t, const Vector3d& f_body, const Vector3d& omega_body) {
    if (!is_initialized_) {
        return;
    }

    const double dt = t - t_;
    if (dt <= 0.0 || dt > 1.0) {
        t_ = t;
        return;
    }

    // Cache latest raw sensor readings for adaptive noise estimation
    last_f_body_ = f_body;
    last_omega_body_ = omega_body;

    // Maintain buffer for stillness detection
    imu_buffer_.push_back({t, f_body, omega_body});
    const double cutoff = t - config_.zupt_accel_mag_window_s;
    auto it = std::remove_if(imu_buffer_.begin(), imu_buffer_.end(),
        [cutoff](const ImuBufferedSample& s) { return s.t < cutoff; });
    imu_buffer_.erase(it, imu_buffer_.end());

    // 1. Unbiased sensor readings
    const Vector3d omega_unbiased = omega_body - b_gyro_;
    const Vector3d f_unbiased = f_body - b_accel_;

    // 2. Attitude propagation via quaternion rotation vector
    const Quaternion dq = Quaternion::fromRotationVector(omega_unbiased * dt);
    q_ = (q_ * dq).normalized();

    // Step 24: Filter signed horizontal yaw rate before taking absolute value to reject rotational vibration
    double w_yaw_signed = 0.0;
    if (config_.nhc_curv_use_nav_yaw) {
        const Vector3d omega_nav = q_.rotate(omega_unbiased);
        w_yaw_signed = omega_nav.z;
    } else {
        w_yaw_signed = omega_unbiased.z;
    }

    if (last_t_lpf_ < 0.0) {
        omega_yaw_filt_ = w_yaw_signed;
    } else if (config_.nhc_curv_lpf_cutoff_hz > 1e-4) {
        const double tau = 1.0 / (2.0 * kPi * config_.nhc_curv_lpf_cutoff_hz);
        const double alpha = dt / (dt + tau);
        omega_yaw_filt_ += alpha * (w_yaw_signed - omega_yaw_filt_);
    } else {
        omega_yaw_filt_ = w_yaw_signed;
    }
    last_t_lpf_ = t;

    // 3. Specific force resolution and gravity subtraction
    const Matrix<3, 3> R = Matrix<3, 3>::fromQuaternion(q_);
    const Vector3d f_nav = q_.rotate(f_unbiased);
    const Vector3d accel_nav = f_nav + Vector3d(0.0, 0.0, -kGravity);

    // 4. Trapezoidal velocity and position propagation
    const Vector3d v_prev = v_enu_;
    const Vector3d delta_v = (last_accel_nav_ + accel_nav) * (0.5 * dt);
    v_enu_ += delta_v;

    const Vector3d delta_p = (v_prev + v_enu_) * (0.5 * dt);
    p_enu_ += delta_p;

    last_accel_nav_ = accel_nav;
    t_ = t;

    // 5. Error state transition matrix Phi = I + F * dt
    Matrix<15, 15> Phi = Matrix<15, 15>::eye();

    // Position kinematics: d(delta_p)/dt = delta_v
    Phi(0, 3) = dt; Phi(1, 4) = dt; Phi(2, 5) = dt;

    // Velocity kinematics: d(delta_v)/dt = -[f_nav]x * delta_theta - R * delta_b_accel
    const Matrix<3, 3> f_skew = Matrix<3, 3>::skewSymmetric(f_nav);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            Phi(3 + r, 6 + c) = -f_skew(r, c) * dt;
            Phi(3 + r, 12 + c) = -R(r, c) * dt;
        }
    }

    // Attitude kinematics: d(delta_theta)/dt = -R * delta_b_gyro
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            Phi(6 + r, 9 + c) = -R(r, c) * dt;
        }
    }

    // 6. Discrete process noise matrix Qd
    Matrix<15, 15> Qd = Matrix<15, 15>::zeros();
    const double q_accel = (config_.sigma_accel_noise * config_.sigma_accel_noise) * dt;
    const double q_gyro = (config_.sigma_gyro_noise * config_.sigma_gyro_noise) * dt;
    const double q_gb = (config_.sigma_gyro_bias * config_.sigma_gyro_bias) * dt;
    const double q_ab = (config_.sigma_accel_bias * config_.sigma_accel_bias) * dt;

    Qd(3, 3) = q_accel; Qd(4, 4) = q_accel; Qd(5, 5) = q_accel;
    Qd(6, 6) = q_gyro;  Qd(7, 7) = q_gyro;  Qd(8, 8) = q_gyro;
    Qd(9, 9) = q_gb;    Qd(10, 10) = q_gb;  Qd(11, 11) = q_gb;
    Qd(12, 12) = q_ab;  Qd(13, 13) = q_ab;  Qd(14, 14) = q_ab;

    // Covariance propagation: P = Phi * P * Phi^T + Qd
    P_ = (Phi * P_ * Phi.transpose() + Qd).symmetrized();
}

bool ErrorStateEkf::isStationary() const {
    if (imu_buffer_.size() < 4) {
        return false;
    }

    // Velocity gate: prevent false detection when moving at highway speeds
    if (v_enu_.norm() > config_.zupt_speed_gate_m_s) {
        return false;
    }

    // Gyroscope magnitude check on unbiased angular rate
    double gyro_norm_sum = 0.0;
    for (const auto& s : imu_buffer_) {
        const Vector3d omega_unbiased = s.omega_body - b_gyro_;
        gyro_norm_sum += omega_unbiased.norm();
    }
    const double mean_gyro_norm = gyro_norm_sum / static_cast<double>(imu_buffer_.size());
    if (mean_gyro_norm > config_.zupt_gyro_thresh_rad_s) {
        return false;
    }

    // Accelerometer magnitude mean and standard deviation
    double accel_norm_sum = 0.0;
    for (const auto& s : imu_buffer_) {
        accel_norm_sum += s.f_body.norm();
    }
    const double mean_accel_norm = accel_norm_sum / static_cast<double>(imu_buffer_.size());

    double var_sum = 0.0;
    for (const auto& s : imu_buffer_) {
        const double diff = s.f_body.norm() - mean_accel_norm;
        var_sum += diff * diff;
    }
    const double accel_std = std::sqrt(var_sum / static_cast<double>(imu_buffer_.size()));
    const double accel_mean_dev = std::abs(mean_accel_norm - kGravity);

    return (accel_std < config_.zupt_accel_std_thresh_m_s2) && (accel_mean_dev < 0.6);
}

bool ErrorStateEkf::updateZupt() {
    // Residual: z = 0 - v_enu = -v_enu
    const Vector3d z = Vector3d(0.0, 0.0, 0.0) - v_enu_;

    // Measurement matrix H: 3x15, mapping velocity delta_v
    // H(0, 3) = 1, H(1, 4) = 1, H(2, 5) = 1
    // Innovation covariance: S = H * P * H^T + R_meas = P[3:6, 3:6] + sigma_zupt^2 * I_3
    Matrix<3, 3> S = P_.getBlock<3, 3>(3, 3);
    const double r_var = config_.sigma_zupt * config_.sigma_zupt;
    S(0, 0) += r_var;
    S(1, 1) += r_var;
    S(2, 2) += r_var;

    Matrix<3, 3> S_inv;
    if (!invert3x3(S, S_inv)) {
        return false;
    }

    // Kalman gain: K = P * H^T * S_inv (15x3)
    // Note: P * H^T is the 3-column block P[0:15, 3:6]
    Matrix<15, 3> P_HT;
    for (size_t r = 0; r < 15; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            P_HT(r, c) = P_(r, 3 + c);
        }
    }
    const Matrix<15, 3> K = P_HT * S_inv;

    // State correction: dx = K * z
    std::array<double, 15> dx{};
    for (size_t r = 0; r < 15; ++r) {
        dx[r] = K(r, 0) * z.x + K(r, 1) * z.y + K(r, 2) * z.z;
    }

    // Nominal state injection
    p_enu_.x += dx[0];
    p_enu_.y += dx[1];
    p_enu_.z += dx[2];

    v_enu_.x += dx[3];
    v_enu_.y += dx[4];
    v_enu_.z += dx[5];

    const Vector3d dtheta(dx[6], dx[7], dx[8]);
    const Quaternion dq = Quaternion::fromRotationVector(dtheta);
    q_ = (dq * q_).normalized();

    b_gyro_.x += dx[9];
    b_gyro_.y += dx[10];
    b_gyro_.z += dx[11];

    b_accel_.x += dx[12];
    b_accel_.y += dx[13];
    b_accel_.z += dx[14];

    // Joseph stabilized covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
    Matrix<15, 15> I_KH = Matrix<15, 15>::eye();
    for (size_t r = 0; r < 15; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            I_KH(r, 3 + c) -= K(r, c);
        }
    }

    Matrix<3, 3> R_meas = Matrix<3, 3>::zeros();
    R_meas(0, 0) = r_var;
    R_meas(1, 1) = r_var;
    R_meas(2, 2) = r_var;

    P_ = (I_KH * P_ * I_KH.transpose() + K * R_meas * K.transpose()).symmetrized();
    return true;
}

double ErrorStateEkf::computeNhcSigmaLat() const noexcept {
    const double dt_init = (t_ > t0_) ? (t_ - t0_) : 0.0;
    double settle_extra = 0.0;
    if (config_.nhc_settle_duration_s > 1e-6) {
        settle_extra = config_.nhc_settle_sigma_extra * std::exp(-dt_init / config_.nhc_settle_duration_s);
    }

    const double speed = getSpeed();
    const double v_curv = config_.nhc_curv_use_speed_floor ? std::max(speed, v0_) : speed;

    double w_yaw = 0.0;
    if (last_t_lpf_ >= 0.0) {
        w_yaw = omega_yaw_filt_;
    } else if (config_.nhc_curv_use_nav_yaw) {
        const Vector3d omega_unbiased = last_omega_body_ - b_gyro_;
        w_yaw = std::abs(q_.rotate(omega_unbiased).z);
    } else {
        w_yaw = std::abs(last_omega_body_.z - b_gyro_.z);
    }

    const double a_c = v_curv * w_yaw;

    const double c_term = config_.nhc_curv_c_coeff * a_c;
    const double yaw_term = config_.nhc_curv_yaw_coeff * w_yaw;

    double lat_term = 0.0;
    if (config_.nhc_curv_lat_coeff > 1e-6) {
        const double a_lat = std::abs(last_f_body_.x - b_accel_.x);
        lat_term = config_.nhc_curv_lat_coeff * a_lat;
    }

    const double curv_scale = std::sqrt(1.0 + c_term * c_term + lat_term * lat_term + yaw_term * yaw_term);

    return (config_.sigma_nhc_lat + settle_extra) * curv_scale;
}

double ErrorStateEkf::computeNhcSigmaVert() const noexcept {
    const double dt_init = (t_ > t0_) ? (t_ - t0_) : 0.0;
    double settle_extra = 0.0;
    if (config_.nhc_settle_duration_s > 1e-6) {
        settle_extra = config_.nhc_settle_sigma_extra * std::exp(-dt_init / config_.nhc_settle_duration_s);
    }
    return config_.sigma_nhc_vert + settle_extra;
}

bool ErrorStateEkf::updateNhc() {
    const Matrix<3, 3> R = Matrix<3, 3>::fromQuaternion(q_);

    // Body velocity: v_b = R^T * v_enu
    // v_b.x is body lateral (Right), v_b.z is body vertical (Up)
    const double v_bx = R(0, 0) * v_enu_.x + R(1, 0) * v_enu_.y + R(2, 0) * v_enu_.z;
    const double v_bz = R(0, 2) * v_enu_.x + R(1, 2) * v_enu_.y + R(2, 2) * v_enu_.z;

    // Measurement residual: z = [0 - v_bx, 0 - v_bz]
    const double z0 = -v_bx;
    const double z1 = -v_bz;

    // Measurement matrix H in R^{2x15}
    // Row 0: lateral velocity constraint
    // Row 1: vertical velocity constraint
    Matrix<2, 15> H = Matrix<2, 15>::zeros();

    // r1 is col 0 of R (row 0 of R^T), r3 is col 2 of R (row 2 of R^T)
    const Vector3d r1(R(0, 0), R(1, 0), R(2, 0));
    const Vector3d r3(R(0, 2), R(1, 2), R(2, 2));

    H(0, 3) = r1.x; H(0, 4) = r1.y; H(0, 5) = r1.z;
    H(1, 3) = r3.x; H(1, 4) = r3.y; H(1, 5) = r3.z;

    const Matrix<3, 3> v_skew = Matrix<3, 3>::skewSymmetric(v_enu_);
    // - (r1 * v_skew)
    H(0, 6) = -(r1.y * v_skew(1, 0) + r1.z * v_skew(2, 0));
    H(0, 7) = -(r1.x * v_skew(0, 1) + r1.z * v_skew(2, 1));
    H(0, 8) = -(r1.x * v_skew(0, 2) + r1.y * v_skew(1, 2));

    H(1, 6) = -(r3.y * v_skew(1, 0) + r3.z * v_skew(2, 0));
    H(1, 7) = -(r3.x * v_skew(0, 1) + r3.z * v_skew(2, 1));
    H(1, 8) = -(r3.x * v_skew(0, 2) + r3.y * v_skew(1, 2));

    // Adaptive innovation covariance S = H * P * H^T + R_meas (2x2)
    Matrix<2, 2> S = H * P_ * H.transpose();
    const double sigma_lat = computeNhcSigmaLat();
    const double sigma_vert = computeNhcSigmaVert();
    const double r_lat = sigma_lat * sigma_lat;
    const double r_vert = sigma_vert * sigma_vert;
    S(0, 0) += r_lat;
    S(1, 1) += r_vert;

    Matrix<2, 2> S_inv;
    if (!invert2x2(S, S_inv)) {
        return false;
    }

    // Kalman gain: K = P * H^T * S_inv (15x2)
    const Matrix<15, 2> K = P_ * H.transpose() * S_inv;

    // State correction: dx = K * z
    std::array<double, 15> dx{};
    for (size_t r = 0; r < 15; ++r) {
        dx[r] = K(r, 0) * z0 + K(r, 1) * z1;
    }

    // Nominal state injection
    p_enu_.x += dx[0];
    p_enu_.y += dx[1];
    p_enu_.z += dx[2];

    v_enu_.x += dx[3];
    v_enu_.y += dx[4];
    v_enu_.z += dx[5];

    const Vector3d dtheta(dx[6], dx[7], dx[8]);
    const Quaternion dq = Quaternion::fromRotationVector(dtheta);
    q_ = (dq * q_).normalized();

    b_gyro_.x += dx[9];
    b_gyro_.y += dx[10];
    b_gyro_.z += dx[11];

    b_accel_.x += dx[12];
    b_accel_.y += dx[13];
    b_accel_.z += dx[14];

    // Joseph stabilized covariance update
    const Matrix<15, 15> I_KH = Matrix<15, 15>::eye() - (K * H);
    Matrix<2, 2> R_meas = Matrix<2, 2>::zeros();
    R_meas(0, 0) = r_lat;
    R_meas(1, 1) = r_vert;

    P_ = (I_KH * P_ * I_KH.transpose() + K * R_meas * K.transpose()).symmetrized();
    return true;
}

bool ErrorStateEkf::update(const ImuSample& sample) {
    predict(
        sample.t,
        Vector3d(sample.ax, sample.ay, sample.az),
        Vector3d(sample.gx, sample.gy, sample.gz)
    );

    if (isStationary()) {
        updateZupt();
    } else if (getSpeed() > 0.5) {
        updateNhc();
    }

    return true;
}

double ErrorStateEkf::getLatitude() const noexcept {
    const double delta_lat_rad = p_enu_.y / kEarthRadius;
    return lat0_ + delta_lat_rad * (180.0 / kPi);
}

double ErrorStateEkf::getLongitude() const noexcept {
    const double cos_lat0 = std::cos(lat0_ * (kPi / 180.0));
    if (std::abs(cos_lat0) < 1e-6) {
        return lon0_;
    }
    const double delta_lon_rad = p_enu_.x / (kEarthRadius * cos_lat0);
    return lon0_ + delta_lon_rad * (180.0 / kPi);
}

double ErrorStateEkf::getAltitude() const noexcept {
    return alt0_ + p_enu_.z;
}

double ErrorStateEkf::getSpeed() const noexcept {
    return std::sqrt(v_enu_.x * v_enu_.x + v_enu_.y * v_enu_.y);
}

double ErrorStateEkf::getHeadingDeg() const noexcept {
    return q_.toHeadingEnu();
}

} // namespace idr
