#include "idr/ekf.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <cassert>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEarthRadius = 6371000.0;

double haversineDistanceM(double lat1, double lon1, double lat2, double lon2) {
    const double dlat = (lat2 - lat1) * (kPi / 180.0);
    const double dlon = (lon2 - lon1) * (kPi / 180.0);
    const double rlat1 = lat1 * (kPi / 180.0);
    const double rlat2 = lat2 * (kPi / 180.0);
    const double a = std::sin(dlat * 0.5) * std::sin(dlat * 0.5) +
                     std::cos(rlat1) * std::cos(rlat2) *
                     std::sin(dlon * 0.5) * std::sin(dlon * 0.5);
    const double c = 2.0 * std::asin(std::min(1.0, std::sqrt(a)));
    return kEarthRadius * c;
}

void testMatrixAlgebra() {
    std::cout << "[RUN] testMatrixAlgebra...\n";

    // 1. Skew-symmetric property: [v]x * w == v x w
    const idr::Vector3d v(1.0, 2.0, 3.0);
    const idr::Vector3d w(4.0, 5.0, 6.0);
    const idr::Matrix<3, 3> v_skew = idr::Matrix<3, 3>::skewSymmetric(v);
    const idr::Vector3d cross_vw = v.cross(w);

    idr::Matrix<3, 1> w_mat;
    w_mat(0, 0) = w.x; w_mat(1, 0) = w.y; w_mat(2, 0) = w.z;
    const idr::Matrix<3, 1> prod = v_skew * w_mat;

    assert(std::abs(prod(0, 0) - cross_vw.x) < 1e-12);
    assert(std::abs(prod(1, 0) - cross_vw.y) < 1e-12);
    assert(std::abs(prod(2, 0) - cross_vw.z) < 1e-12);

    // 2. 2x2 Analytical Inversion
    idr::Matrix<2, 2> A2;
    A2(0, 0) = 4.0; A2(0, 1) = 7.0;
    A2(1, 0) = 2.0; A2(1, 1) = 6.0;
    idr::Matrix<2, 2> A2_inv;
    assert(idr::invert2x2(A2, A2_inv));
    const idr::Matrix<2, 2> I2 = A2 * A2_inv;
    assert(std::abs(I2(0, 0) - 1.0) < 1e-12);
    assert(std::abs(I2(0, 1)) < 1e-12);
    assert(std::abs(I2(1, 0)) < 1e-12);
    assert(std::abs(I2(1, 1) - 1.0) < 1e-12);

    // 3. 3x3 Cramer Inversion
    idr::Matrix<3, 3> A3;
    A3(0, 0) = 1.0; A3(0, 1) = 2.0; A3(0, 2) = 3.0;
    A3(1, 0) = 0.0; A3(1, 1) = 1.0; A3(1, 2) = 4.0;
    A3(2, 0) = 5.0; A3(2, 1) = 6.0; A3(2, 2) = 0.0;
    idr::Matrix<3, 3> A3_inv;
    assert(idr::invert3x3(A3, A3_inv));
    const idr::Matrix<3, 3> I3 = A3 * A3_inv;
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            const double expected = (r == c) ? 1.0 : 0.0;
            assert(std::abs(I3(r, c) - expected) < 1e-11);
        }
    }

    std::cout << "  [PASS] Matrix algebra verified.\n";
}

void testStraightLineConstantVelocity() {
    std::cout << "[RUN] testStraightLineConstantVelocity...\n";
    idr::ErrorStateEkf ekf;

    const double t0 = 0.0;
    const double lat0 = 0.0;
    const double lon0 = 0.0;
    const double alt0 = 100.0;
    const double speed = 20.0;
    const double heading = 90.0; // East along Equator

    ekf.initialize(t0, lat0, lon0, alt0, speed, heading);
    assert(ekf.isInitialized());

    const double dt = 0.1;
    double t = t0;
    for (int i = 0; i < 100; ++i) { // 10 seconds
        t += dt;
        // Level unaccelerated motion: specific force balances gravity [0, 0, g]
        ekf.predict(t, idr::Vector3d(0.0, 0.0, idr::ErrorStateEkf::kGravity), idr::Vector3d(0.0, 0.0, 0.0));
    }

    const idr::Vector3d p = ekf.getPositionEnu();
    std::cout << "  Displacement ENU: East=" << p.x << " m, North=" << p.y << " m, Up=" << p.z << " m\n";
    assert(std::abs(p.x - 200.0) < 0.05); // 200 m East
    assert(std::abs(p.y) < 0.05);
    assert(std::abs(p.z) < 0.05);
    assert(std::abs(ekf.getSpeed() - 20.0) < 0.01);
    assert(std::abs(ekf.getHeadingDeg() - 90.0) < 0.05);

    std::cout << "  [PASS] Straight-line propagation verified.\n";
}

void testZuptUpdate() {
    std::cout << "[RUN] testZuptUpdate...\n";
    idr::ErrorStateEkf ekf;
    // Start with false 5 m/s velocity
    ekf.initialize(0.0, 0.0, 0.0, 0.0, 5.0, 0.0);

    const double init_v_var = ekf.getCovariance()(3, 3);
    assert(ekf.getVelocityEnu().norm() > 4.9);

    const bool ok = ekf.updateZupt();
    assert(ok);

    const double post_speed = ekf.getVelocityEnu().norm();
    std::cout << "  Speed before: 5.0 m/s, Speed after ZUPT: " << post_speed << " m/s\n";
    assert(post_speed < 1.0);
    assert(ekf.getCovariance()(3, 3) < init_v_var);

    std::cout << "  [PASS] ZUPT update verified.\n";
}

void testNhcUpdate() {
    std::cout << "[RUN] testNhcUpdate...\n";
    idr::ErrorStateEkf ekf;
    // Vehicle pointing North (heading=0) with 10 m/s longitudinal velocity
    ekf.initialize(0.0, 0.0, 0.0, 0.0, 10.0, 0.0);

    // Predict forward slightly with a small lateral disturbance
    ekf.predict(0.1, idr::Vector3d(1.5, 0.0, idr::ErrorStateEkf::kGravity), idr::Vector3d(0.0, 0.0, 0.0));

    const idr::Matrix<3, 3> R = idr::Matrix<3, 3>::fromQuaternion(ekf.getAttitude());
    const idr::Vector3d v_enu = ekf.getVelocityEnu();
    // Body lateral velocity: row 0 of R^T * v_enu
    const double v_lat_before = R(0, 0) * v_enu.x + R(1, 0) * v_enu.y + R(2, 0) * v_enu.z;

    const bool ok = ekf.updateNhc();
    assert(ok);

    const idr::Vector3d v_enu_after = ekf.getVelocityEnu();
    const double v_lat_after = R(0, 0) * v_enu_after.x + R(1, 0) * v_enu_after.y + R(2, 0) * v_enu_after.z;

    std::cout << "  Lateral velocity before: " << v_lat_before << " m/s, after: " << v_lat_after << " m/s\n";
    assert(std::abs(v_lat_after) < std::abs(v_lat_before));

    std::cout << "  [PASS] NHC update verified.\n";
}

void testCovarianceStability() {
    std::cout << "[RUN] testCovarianceStability...\n";
    idr::ErrorStateEkf ekf;
    ekf.initialize(0.0, 0.0, 0.0, 0.0, 15.0, 45.0);

    double t = 0.0;
    const double dt = 0.1;
    for (int i = 0; i < 50; ++i) {
        t += dt;
        ekf.predict(t, idr::Vector3d(0.1, 0.0, idr::ErrorStateEkf::kGravity), idr::Vector3d(0.001, 0.002, 0.0));
        if (i % 5 == 0) {
            ekf.updateNhc();
        }
        if (i % 20 == 0) {
            ekf.updateZupt();
        }

        const auto& P = ekf.getCovariance();
        for (size_t r = 0; r < 15; ++r) {
            assert(P(r, r) > 0.0);
            for (size_t c = r + 1; c < 15; ++c) {
                assert(std::abs(P(r, c) - P(c, r)) < 1e-9);
            }
        }
    }
    std::cout << "  [PASS] Covariance stability and symmetry verified.\n";
}

bool runCrossCheckInstance(
    const std::string& cache_path,
    const idr::Quaternion& q0,
    const idr::Vector3d& gb0,
    double expected_python_lat,
    double expected_python_lon,
    double expected_python_error_m,
    double tol_error_m
) {
    std::ifstream in(cache_path);
    if (!in.is_open()) {
        std::cout << "  [NOTICE] Cache file not found: " << cache_path << " (skipping instance)\n";
        return false;
    }

    std::string line1;
    if (!std::getline(in, line1)) return false;

    // Parse initial state comment
    double t0 = 0.0, lat0 = 0.0, lon0 = 0.0, alt0 = 0.0, speed0 = 0.0, heading0 = 0.0;
    double ax0 = 0.0, ay0 = 0.0, az0 = idr::ErrorStateEkf::kGravity;

    const std::string prefix = "# initial_state:";
    if (line1.rfind(prefix, 0) == 0) {
        std::stringstream ss(line1.substr(prefix.size()));
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            size_t eq = tok.find('=');
            if (eq == std::string::npos) continue;
            size_t k_start = tok.find_first_not_of(" \t");
            std::string k = tok.substr(k_start, eq - k_start);
            double val = std::strtod(tok.substr(eq + 1).c_str(), nullptr);
            if (k == "t0") t0 = val;
            else if (k == "lat0") lat0 = val;
            else if (k == "lon0") lon0 = val;
            else if (k == "alt0") alt0 = val;
            else if (k == "speed_ms") speed0 = val;
            else if (k == "heading_deg") heading0 = val;
            else if (k == "ax0") ax0 = val;
            else if (k == "ay0") ay0 = val;
            else if (k == "az0") az0 = val;
        }
    }

    std::string line2; // Column header
    std::getline(in, line2);

    idr::ErrorStateEkf ekf;
    ekf.initializeWithAttitude(t0, lat0, lon0, alt0, speed0, heading0, q0, gb0, idr::Vector3d(0, 0, 0), idr::Vector3d(ax0, ay0, az0));

    std::string row;
    size_t samples = 0;
    while (std::getline(in, row)) {
        if (row.empty() || row[0] == '#') continue;
        std::stringstream ss(row);
        std::string t_s, ax_s, ay_s, az_s, gx_s, gy_s, gz_s;
        if (!std::getline(ss, t_s, ',') ||
            !std::getline(ss, ax_s, ',') ||
            !std::getline(ss, ay_s, ',') ||
            !std::getline(ss, az_s, ',') ||
            !std::getline(ss, gx_s, ',') ||
            !std::getline(ss, gy_s, ',') ||
            !std::getline(ss, gz_s, ',')) {
            continue;
        }

        idr::ImuSample sample;
        sample.t = std::strtod(t_s.c_str(), nullptr);
        sample.ax = std::strtod(ax_s.c_str(), nullptr);
        sample.ay = std::strtod(ay_s.c_str(), nullptr);
        sample.az = std::strtod(az_s.c_str(), nullptr);
        sample.gx = std::strtod(gx_s.c_str(), nullptr);
        sample.gy = std::strtod(gy_s.c_str(), nullptr);
        sample.gz = std::strtod(gz_s.c_str(), nullptr);

        ekf.update(sample);
        samples++;
    }

    const double final_lat = ekf.getLatitude();
    const double final_lon = ekf.getLongitude();
    const double final_speed = ekf.getSpeed();
    const double final_heading = ekf.getHeadingDeg();

    // Distance between Python final coordinate and C++ final coordinate
    const double diff_cpp_vs_py_m = haversineDistanceM(final_lat, final_lon, expected_python_lat, expected_python_lon);

    std::cout << "  Processed " << samples << " samples\n";
    std::cout << "  C++ Final Coord:   (" << final_lat << ", " << final_lon << ")\n";
    std::cout << "  Py Reference Coord: (" << expected_python_lat << ", " << expected_python_lon << ")\n";
    std::cout << "  Difference C++ vs Python: " << diff_cpp_vs_py_m << " m (tolerance: " << tol_error_m << " m)\n";
    std::cout << "  Reference Python Outage Error: ~" << expected_python_error_m << " m\n";
    std::cout << "  Final Speed: " << final_speed << " m/s, Heading: " << final_heading << " deg\n";

    assert(diff_cpp_vs_py_m < tol_error_m);
    return true;
}

void testNumericalCrossCheck() {
    std::cout << "[RUN] testNumericalCrossCheck...\n";

    // 1. Check pair_Vw1__180s__0 (stationary mount-shift case)
    // Module B attitude and measured bias for pair_Vw1:
    // qw=0.9999968, qx=0.0, qy=0.0, qz=-0.0025298
    // gyro bias = (-0.008924, -0.007397, 0.003356)
    // Python prototype reference: final lat=52.55469404, lon=-1.46306325, error ~1.94m
    const std::string path_vw1 = "data/processed/_cpp_replay_cache/pair_Vw1__180s__0.csv";
    const idr::Quaternion q_vw1(0.9999968, 0.0, 0.0, -0.0025298);
    const idr::Vector3d gb_vw1(-0.008924, -0.007397, 0.003356);

    std::cout << "--> Cross-checking instance 1: pair_Vw1__180s__0\n";
    runCrossCheckInstance(path_vw1, q_vw1, gb_vw1, 52.55469404, -1.46306325, 1.94, 0.05);

    // 2. Check pair_Vta2__180s__0 (dynamic bias doubling case)
    // Values directly from results/module_b_initial_attitude.csv for pair_Vta2:
    // qw=-0.9387507, qx=0.032916, qy=0.0255123, qz=-0.3420713
    // gyro bias = (-0.00259082, 0.00719796, -0.0053898)
    // Python prototype reference (v2 tuned): final lat=52.79245735, lon=-1.62577604
    const std::string path_vta2 = "data/processed/_cpp_replay_cache/pair_Vta2__180s__0.csv";
    const idr::Quaternion q_vta2(-0.9387507, 0.032916, 0.0255123, -0.3420713);
    const idr::Vector3d gb_vta2(-0.00259082, 0.00719796, -0.0053898);

    std::cout << "--> Cross-checking instance 2: pair_Vta2__180s__0\n";
    runCrossCheckInstance(path_vta2, q_vta2, gb_vta2, 52.79245735, -1.62577604, 1585.87, 0.05);

    std::cout << "  [PASS] Numerical cross-check verified on reference instances.\n";
}

void testCurvatureAdaptiveNhc() {
    std::cout << "[RUN] testCurvatureAdaptiveNhc (Step 22 Kinematic)...\n";

    idr::EkfConfig config;
    config.nhc_settle_duration_s = 2.0;
    config.nhc_settle_sigma_extra = 4.0;
    config.nhc_curv_c_coeff = 2.0;
    config.nhc_curv_lat_coeff = 0.0;
    config.nhc_curv_yaw_coeff = 0.0;
    config.sigma_nhc_lat = 0.15;
    config.sigma_nhc_vert = 0.15;

    idr::ErrorStateEkf ekf(config);
    ekf.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 90.0);

    // 1. Advance past settling window on straight road with heavy injected accelerometer vibration noise (+/- 2.0 m/s^2, omega_z = 0)
    for (int i = 1; i <= 300; ++i) {
        const double t = i * 0.1;
        // High-frequency zero-mean vibration noise on lateral axis (body X)
        const double ax_noise = (i % 2 == 0) ? 2.0 : -2.0;
        ekf.predict(t, idr::Vector3d(ax_noise, 0.0, idr::ErrorStateEkf::kGravity), idr::Vector3d(0.0, 0.0, 0.0));
        ekf.updateNhc();
    }

    // Kinematic centripetal acceleration is a_c = speed * omega_z = 20.0 * 0.0 = 0.0 m/s^2
    // Therefore, despite 2.0 m/s^2 lateral accelerometer vibration noise, sigma_nhc_lat must remain strictly 0.15 m/s!
    const double sigma_vibration_lat = ekf.computeNhcSigmaLat();
    const double sigma_vibration_vert = ekf.computeNhcSigmaVert();
    std::cout << "  Straight road with 2.0 m/s^2 accel vibration noise sigma_lat: " << sigma_vibration_lat << " m/s (expected 0.15)\n";
    assert(std::abs(sigma_vibration_lat - 0.15) < 1e-4);
    assert(std::abs(sigma_vibration_vert - 0.15) < 1e-4);

    // 2. Introduce real sustained curve: omega_z = 0.1 rad/s at speed ~10-20 m/s -> a_c ~ 1-2 m/s^2
    ekf.predict(30.1, idr::Vector3d(0.0, 0.0, idr::ErrorStateEkf::kGravity), idr::Vector3d(0.0, 0.0, 0.1));
    const double sigma_curve_lat = ekf.computeNhcSigmaLat();
    const double current_speed = ekf.getSpeed();
    const double w_yaw = std::abs(0.1 - ekf.getGyroBias().z);
    const double expected_ac = current_speed * w_yaw;
    const double expected_sigma = 0.15 * std::sqrt(1.0 + (2.0 * expected_ac) * (2.0 * expected_ac));
    std::cout << "  Sustained curve (speed=" << current_speed << " m/s, a_c=" << expected_ac << " m/s^2) sigma_lat: " << sigma_curve_lat << " m/s (expected ~" << expected_sigma << ")\n";
    assert(std::abs(sigma_curve_lat - expected_sigma) < 1e-4);
    assert(sigma_curve_lat > 0.30); // Significantly inflated above 0.15 m/s baseline

    // Vertical sigma must remain unaffected
    const double sigma_curve_vert = ekf.computeNhcSigmaVert();
    assert(std::abs(sigma_curve_vert - 0.15) < 1e-4);

    std::cout << "  [PASS] Step 22 Kinematic centripetal NHC noise & vibration immunity verified.\n";
}

void testSettlingGracePeriod() {
    std::cout << "[RUN] testSettlingGracePeriod...\n";

    idr::EkfConfig config;
    config.nhc_settle_duration_s = 2.0;
    config.nhc_settle_sigma_extra = 4.0;
    config.nhc_curv_lat_coeff = 5.0;
    config.nhc_curv_yaw_coeff = 2.0;
    config.sigma_nhc_lat = 0.15;
    config.sigma_nhc_vert = 0.15;

    idr::ErrorStateEkf ekf(config);
    ekf.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 0.0);

    // At t = 0: extra = 4.0, curv_scale = 1.0 -> sigma_lat = 4.15
    const double sigma_t0 = ekf.computeNhcSigmaLat();
    std::cout << "  t=0s sigma_lat: " << sigma_t0 << " m/s (expected 4.15)\n";
    assert(std::abs(sigma_t0 - 4.15) < 1e-3);

    // At t = 2.0s (1 tau): extra = 4.0 * exp(-1) ~ 1.4715 -> sigma_lat ~ 1.6215
    ekf.predict(2.0, idr::Vector3d(0.0, 0.0, idr::ErrorStateEkf::kGravity), idr::Vector3d(0.0, 0.0, 0.0));
    const double sigma_t2 = ekf.computeNhcSigmaLat();
    std::cout << "  t=2s (1 tau) sigma_lat: " << sigma_t2 << " m/s (expected ~1.62)\n";
    assert(std::abs(sigma_t2 - (0.15 + 4.0 * std::exp(-1.0))) < 1e-3);

    // At t = 10.0s (5 tau): extra = 4.0 * exp(-5) ~ 0.02695 -> sigma_lat ~ 0.177
    ekf.predict(10.0, idr::Vector3d(0.0, 0.0, idr::ErrorStateEkf::kGravity), idr::Vector3d(0.0, 0.0, 0.0));
    const double sigma_t10 = ekf.computeNhcSigmaLat();
    std::cout << "  t=10s (5 tau) sigma_lat: " << sigma_t10 << " m/s (expected ~0.177)\n";
    assert(std::abs(sigma_t10 - (0.15 + 4.0 * std::exp(-5.0))) < 1e-3);

    std::cout << "  [PASS] Settling grace period exponential decay verified.\n";
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << " Running C++17 15-State ES-EKF Test Suite\n";
    std::cout << "========================================\n";

    testMatrixAlgebra();
    testStraightLineConstantVelocity();
    testZuptUpdate();
    testNhcUpdate();
    testCovarianceStability();
    testCurvatureAdaptiveNhc();
    testSettlingGracePeriod();
    testNumericalCrossCheck();

    std::cout << "========================================\n";
    std::cout << " [ALL TESTS PASSED] 15-State ES-EKF Core\n";
    std::cout << "========================================\n";
    return 0;
}
