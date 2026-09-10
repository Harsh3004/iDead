#include "idr/strapdown.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

namespace {

constexpr double kPi = 3.14159265358979323846;

void testStraightLineConstantVelocity() {
    std::cout << "[RUN] testStraightLineConstantVelocity...\n";
    idr::StrapdownIns ins;

    const double t0 = 0.0;
    const double lat0 = 0.0;
    const double lon0 = 0.0;
    const double alt0 = 100.0;
    const double speed = 20.0; // 20 m/s
    const double heading = 90.0; // Heading due East along Equator

    // At rest or steady level motion, accelerometer measures +g upward in body Z
    const idr::Vector3d initial_accel(0.0, 0.0, idr::StrapdownIns::kGravity);
    ins.initialize(t0, lat0, lon0, alt0, speed, heading, initial_accel);

    assert(ins.isInitialized());
    assert(std::abs(ins.getSpeed() - 20.0) < 1e-6);
    assert(std::abs(ins.getHeadingDeg() - 90.0) < 1e-5);

    // Propagate 10 seconds at 20 Hz (200 steps of dt = 0.05s)
    const double dt = 0.05;
    const int num_steps = 200;
    double t = t0;

    for (int i = 1; i <= num_steps; ++i) {
        t += dt;
        idr::ImuSample sample{
            t,
            0.0, 0.0, idr::StrapdownIns::kGravity, // ax, ay, az (level, gravity reaction only)
            0.0, 0.0, 0.0,                        // gx, gy, gz (zero rotation)
            0.0, 0.0, 0.0                         // mx, my, mz
        };
        ins.update(sample);
    }

    const idr::Vector3d pos_enu = ins.getPositionEnu();
    const double expected_dist = speed * (t - t0); // 200.0 m East

    // Check ENU displacements
    std::cout << "  East displacement: " << pos_enu.x << " m (expected: " << expected_dist << " m)\n";
    std::cout << "  North displacement: " << pos_enu.y << " m (expected: 0.0 m)\n";
    std::cout << "  Up displacement: " << pos_enu.z << " m (expected: 0.0 m)\n";

    assert(std::abs(pos_enu.x - expected_dist) < 0.01); // < 1 cm error
    assert(std::abs(pos_enu.y) < 0.01);
    assert(std::abs(pos_enu.z) < 0.01);

    // Check geodetic coordinates along equator
    const double expected_lon_deg = (expected_dist / idr::StrapdownIns::kEarthRadius) * (180.0 / kPi);
    assert(std::abs(ins.getLatitude() - lat0) < 1e-6);
    assert(std::abs(ins.getLongitude() - expected_lon_deg) < 1e-6);
    assert(std::abs(ins.getAltitude() - alt0) < 1e-4);

    std::cout << "  [PASS] testStraightLineConstantVelocity succeeded.\n";
}

void testCircularArcTurn() {
    std::cout << "[RUN] testCircularArcTurn...\n";
    idr::StrapdownIns ins;

    const double t0 = 0.0;
    const double lat0 = 12.9716;
    const double lon0 = 77.5946;
    const double alt0 = 900.0;
    const double speed = 10.0;   // 10 m/s
    const double heading = 0.0;  // Heading North (0 deg)

    // Circular turn parameters:
    // Radius R = 100 m, Turn rate omega = v / R = 0.1 rad/s (counter-clockwise turn)
    // Centripetal acceleration in body frame: a_x = -omega * v = -1.0 m/s^2 (towards center on left)
    const double radius = 100.0;
    const double omega_z = speed / radius; // 0.1 rad/s
    const double centripetal_accel_x = -omega_z * speed; // -1.0 m/s^2

    const idr::Vector3d initial_accel(centripetal_accel_x, 0.0, idr::StrapdownIns::kGravity);
    ins.initialize(t0, lat0, lon0, alt0, speed, heading, initial_accel);

    // Quarter circle turn: angle = pi / 2 = 90 deg turn
    const double duration = (kPi / 2.0) / omega_z; // ~15.70796 s
    const double dt = 0.01; // 100 Hz simulation
    const int num_steps = static_cast<int>(std::round(duration / dt));

    double t = t0;
    for (int i = 1; i <= num_steps; ++i) {
        t += dt;
        idr::ImuSample sample{
            t,
            centripetal_accel_x, 0.0, idr::StrapdownIns::kGravity,
            0.0, 0.0, omega_z,
            0.0, 0.0, 0.0
        };
        ins.update(sample);
    }

    // Analytic closed-form solution for 90-deg left turn starting North:
    // Vehicle starts pointing North, turns CCW (West).
    // Center of circle is at (-R, 0) in ENU.
    // Circle equation: (x + R)^2 + y^2 = R^2
    // Angle traversed: theta = omega_z * t = pi / 2
    // Position: East = -R + R * cos(theta) = -R + 0 = -100 m
    //           North = R * sin(theta) = 100 m
    const double expected_east = -radius + radius * std::cos(omega_z * (t - t0));
    const double expected_north = radius * std::sin(omega_z * (t - t0));

    const idr::Vector3d actual_pos = ins.getPositionEnu();
    std::cout << "  Simulated ENU: (" << actual_pos.x << ", " << actual_pos.y << ")\n";
    std::cout << "  Analytic ENU:  (" << expected_east << ", " << expected_north << ")\n";

    const double pos_error = std::sqrt(
        (actual_pos.x - expected_east) * (actual_pos.x - expected_east) +
        (actual_pos.y - expected_north) * (actual_pos.y - expected_north)
    );
    std::cout << "  Circular arc trajectory error: " << pos_error << " m (over " << radius << "m radius turn)\n";

    // Strapdown integration should reproduce the circular path accurately (< 0.2 m error over 15.7s)
    assert(pos_error < 0.2);

    // Verify heading turned 90 deg CCW from 0 deg North to 270 deg West
    const double actual_heading = ins.getHeadingDeg();
    std::cout << "  End heading: " << actual_heading << " deg (expected: ~270.0 deg)\n";
    assert(std::abs(actual_heading - 270.0) < 0.5);

    // Verify it is definitively curved and not a straight tangent line (which would be at (0, 157.1))
    assert(actual_pos.x < -90.0); // Has moved strongly westward into the turn

    std::cout << "  [PASS] testCircularArcTurn succeeded.\n";
}

void testGyroBiasErrorCharacterization() {
    std::cout << "[RUN] testGyroBiasErrorCharacterization...\n";

    // Characterization test: verify that uncorrected gyro bias causes cubic position error growth
    // and that error is strictly monotonically increasing with outage duration.
    // An uncorrected pitch/roll gyro bias tilts the estimated attitude, causing gravity
    // to leak into horizontal acceleration (a_nav = g * sin(delta_theta) ~ g * omega * t),
    // producing velocity error ~ 0.5 * g * omega * t^2 and position error ~ (1/6) * g * omega * t^3.
    const double gyro_bias_pitch = 0.001; // ~0.057 deg/s uncorrected bias
    const double speed = 15.0;            // 15 m/s

    const std::vector<double> test_durations = {10.0, 30.0, 60.0, 120.0};
    std::vector<double> errors;

    for (double duration : test_durations) {
        idr::StrapdownIns ins;
        const idr::Vector3d initial_accel(0.0, 0.0, idr::StrapdownIns::kGravity);
        ins.initialize(0.0, 0.0, 0.0, 0.0, speed, 0.0, initial_accel);

        const double dt = 0.05;
        const int steps = static_cast<int>(std::round(duration / dt));
        double t = 0.0;

        for (int i = 1; i <= steps; ++i) {
            t += dt;
            idr::ImuSample sample{
                t,
                0.0, 0.0, idr::StrapdownIns::kGravity,
                gyro_bias_pitch, 0.0, 0.0, // Pitch gyro bias
                0.0, 0.0, 0.0
            };
            ins.update(sample);
        }

        // True position without bias: straight North (0, speed * duration, 0)
        const double true_north = speed * duration;
        const idr::Vector3d actual_pos = ins.getPositionEnu();

        const double err = std::sqrt(
            actual_pos.x * actual_pos.x +
            (actual_pos.y - true_north) * (actual_pos.y - true_north)
        );
        errors.push_back(err);
        std::cout << "  Duration " << duration << "s -> Position Error: " << err << " m\n";
    }

    // Assert strictly monotonic error growth: err(10s) < err(30s) < err(60s) < err(120s)
    for (size_t i = 1; i < errors.size(); ++i) {
        assert(errors[i] > errors[i - 1]);
    }

    // Verify cubic-like growth: error at 60s should be substantially greater than 8x error at 30s
    // (since (60/30)^3 = 8)
    std::cout << "  Ratio err(60s) / err(30s) = " << (errors[2] / errors[1]) << " (cubic scaling predicts ~8.0)\n";
    assert(errors[2] / errors[1] > 5.0);

    std::cout << "  [PASS] testGyroBiasErrorCharacterization succeeded.\n";
}

void testAttitudeAndBiasInjection() {
    std::cout << "[RUN] testAttitudeAndBiasInjection...\n";

    // 1. Verify gyro bias cancellation:
    // With gyro bias injected, a matching constant gyro measurement should be completely cancelled,
    // preventing the cubic position error blowup seen in testGyroBiasErrorCharacterization.
    const double gyro_bias_pitch = 0.001; // rad/s
    const double speed = 15.0;            // m/s
    const double duration = 60.0;         // 60s outage

    idr::StrapdownIns ins;
    const idr::Vector3d initial_accel(0.0, 0.0, idr::StrapdownIns::kGravity);
    const idr::Quaternion q0 = idr::Quaternion::fromHeadingEnu(0.0);
    const idr::Vector3d bias(gyro_bias_pitch, 0.0, 0.0);

    ins.initializeWithAttitude(0.0, 0.0, 0.0, 0.0, speed, 0.0, q0, bias, initial_accel);
    assert(ins.isInitialized());
    assert(std::abs(ins.getGyroBias().x - gyro_bias_pitch) < 1e-12);

    const double dt = 0.05;
    const int steps = static_cast<int>(std::round(duration / dt));
    double t = 0.0;

    for (int i = 1; i <= steps; ++i) {
        t += dt;
        idr::ImuSample sample{
            t,
            0.0, 0.0, idr::StrapdownIns::kGravity,
            gyro_bias_pitch, 0.0, 0.0, // IMU measures bias
            0.0, 0.0, 0.0
        };
        ins.update(sample);
    }

    const double true_north = speed * duration;
    const idr::Vector3d actual_pos = ins.getPositionEnu();
    const double err = std::sqrt(
        actual_pos.x * actual_pos.x +
        (actual_pos.y - true_north) * (actual_pos.y - true_north)
    );

    std::cout << "  60s Position Error with bias cancellation: " << err << " m\n";
    // Without bias cancellation, 60s error was > 100 meters. With cancellation, it must be < 0.01m
    assert(err < 0.01);

    // 2. Verify pitch/roll leveling cancellation:
    // A tilted vehicle on an incline (pitch = 3 deg) has specific force rotated in body frame:
    // f_body = [0, -g * sin(pitch), g * cos(pitch)]
    const double pitch_rad = 3.0 * (kPi / 180.0);
    const double roll_rad = -2.0 * (kPi / 180.0);
    const double heading_deg = 45.0;

    const idr::Quaternion q_tilted = idr::Quaternion::fromEulerEnu(heading_deg, pitch_rad, roll_rad);
    // In still conditions, specific force in ENU is [0, 0, g]. Body accelerometer reads R^T * [0, 0, g]:
    const idr::Vector3d f_body = q_tilted.conjugate().rotate(idr::Vector3d(0.0, 0.0, idr::StrapdownIns::kGravity));

    idr::StrapdownIns ins_tilted;
    ins_tilted.initializeWithAttitude(0.0, 0.0, 0.0, 0.0, 0.0, heading_deg, q_tilted, idr::Vector3d(0.0, 0.0, 0.0), f_body);

    // Initial kinematic acceleration must be 0
    const idr::Vector3d init_accel_enu = ins_tilted.getAccelerationEnu();
    assert(init_accel_enu.norm() < 1e-6);

    // Update 5 seconds at rest
    for (int i = 1; i <= 100; ++i) {
        idr::ImuSample sample{
            0.05 * i,
            f_body.x, f_body.y, f_body.z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        };
        ins_tilted.update(sample);
    }

    const idr::Vector3d tilted_pos = ins_tilted.getPositionEnu();
    std::cout << "  5s Stationary drift on incline: " << tilted_pos.norm() << " m\n";
    assert(tilted_pos.norm() < 1e-6);

    std::cout << "  [PASS] testAttitudeAndBiasInjection succeeded.\n";
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << " Running Strapdown INS Synthetic Unit Tests\n";
    std::cout << "========================================\n";

    testStraightLineConstantVelocity();
    testCircularArcTurn();
    testGyroBiasErrorCharacterization();
    testAttitudeAndBiasInjection();

    std::cout << "========================================\n";
    std::cout << " [ALL PASSED] All Strapdown INS tests passed!\n";
    std::cout << "========================================\n";
    return 0;
}
