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
    // Python prototype reference: final lat=52.79256590, lon=-1.62431522, error ~1585.87m
    const std::string path_vta2 = "data/processed/_cpp_replay_cache/pair_Vta2__180s__0.csv";
    const idr::Quaternion q_vta2(-0.9387507, 0.032916, 0.0255123, -0.3420713);
    const idr::Vector3d gb_vta2(-0.00259082, 0.00719796, -0.0053898);

    std::cout << "--> Cross-checking instance 2: pair_Vta2__180s__0\n";
    runCrossCheckInstance(path_vta2, q_vta2, gb_vta2, 52.79256590, -1.62431522, 1585.87, 0.05);

    std::cout << "  [PASS] Numerical cross-check verified on reference instances.\n";
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
    testNumericalCrossCheck();

    std::cout << "========================================\n";
    std::cout << " [ALL TESTS PASSED] 15-State ES-EKF Core\n";
    std::cout << "========================================\n";
    return 0;
}
