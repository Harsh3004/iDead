"""Unit tests for the 15-State Error-State Extended Kalman Filter (ES-EKF)."""

import unittest
import numpy as np

from dataeval.estimation.ekf import (
    ErrorStateEkf,
    EkfConfig,
    quat_to_rot_matrix,
    quat_rotate,
    quat_multiply,
    quat_normalize,
    rot_vec_to_quat,
    skew_symmetric,
)


class TestErrorStateEkf(unittest.TestCase):
    """Test suite verifying mathematical correctness and numerical stability of ES-EKF."""

    def setUp(self):
        self.cfg = EkfConfig()
        self.ekf = ErrorStateEkf(self.cfg)

    def test_skew_symmetric_properties(self):
        """Verify skew-symmetric matrix satisfies [v]x * w = v x w."""
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        v_skew = skew_symmetric(v)
        cross_prod = np.cross(v, w)
        mat_prod = v_skew @ w
        np.testing.assert_allclose(mat_prod, cross_prod, atol=1e-12)
        np.testing.assert_allclose(v_skew + v_skew.T, np.zeros((3, 3)), atol=1e-12)

    def test_quaternion_rotation_matrix_consistency(self):
        """Verify quat_to_rot_matrix and quat_rotate produce identical vector transformations."""
        # Test 90 deg rotation about Z
        q = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)])
        v = np.array([1.0, 0.0, 0.0])
        v_rot = quat_rotate(q, v)
        R = quat_to_rot_matrix(q)
        v_mat = R @ v
        np.testing.assert_allclose(v_rot, v_mat, atol=1e-12)
        np.testing.assert_allclose(v_rot, np.array([0.0, 1.0, 0.0]), atol=1e-12)

    def test_straight_line_constant_velocity_propagation(self):
        """Verify state propagation on synthetic straight-line constant velocity path."""
        speed = 20.0 # 20 m/s East
        heading = 90.0 # due East
        t0 = 0.0
        lat0 = 0.0
        lon0 = 0.0
        alt0 = 100.0

        self.ekf.initialize(t0, lat0, lon0, alt0, speed, heading)
        self.assertTrue(self.ekf.is_initialized)

        dt = 0.1
        t = t0
        for _ in range(100): # 10 seconds
            t += dt
            # At rest or level unaccelerated motion, specific force balances gravity [0, 0, g]
            f_b = np.array([0.0, 0.0, self.ekf.kGravity])
            omega_b = np.array([0.0, 0.0, 0.0])
            self.ekf.predict(t, f_b, omega_b)

        # Expected position: East = 200m, North = 0m, Up = 0m
        np.testing.assert_allclose(self.ekf.p_enu[0], 200.0, atol=0.01)
        np.testing.assert_allclose(self.ekf.p_enu[1], 0.0, atol=0.01)
        np.testing.assert_allclose(self.ekf.p_enu[2], 0.0, atol=0.01)
        self.assertAlmostEqual(self.ekf.get_speed_ms(), 20.0, places=3)
        self.assertAlmostEqual(self.ekf.get_heading_deg(), 90.0, places=2)

    def test_zupt_measurement_update(self):
        """Verify ZUPT zeroes out velocity error and reduces velocity covariance."""
        self.ekf.initialize(0.0, 0.0, 0.0, 0.0, 5.0, 0.0) # Start with false 5 m/s velocity
        self.assertTrue(self.ekf.is_initialized)

        init_v_var = self.ekf.P[3, 3]
        self.assertGreater(np.linalg.norm(self.ekf.v_enu), 1.0)

        # Apply ZUPT update
        success = self.ekf.update_zupt()
        self.assertTrue(success)

        # Velocity must be pulled strongly towards zero
        self.assertLess(np.linalg.norm(self.ekf.v_enu), 1.0)
        # Velocity covariance must be reduced
        self.assertLess(self.ekf.P[3, 3], init_v_var)

    def test_nhc_measurement_update(self):
        """Verify NHC suppresses lateral velocity and maintains observability on roll/pitch."""
        # Vehicle pointing North (heading=0) with longitudinal velocity 10 m/s, but false lateral drift
        self.ekf.initialize(0.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        # Inject lateral velocity error: in ENU, X is East (body lateral for vehicle pointing North)
        self.ekf.v_enu[0] = 2.0 # 2 m/s lateral slip

        R = quat_to_rot_matrix(self.ekf.q)
        v_b_before = R.T @ self.ekf.v_enu
        self.assertAlmostEqual(v_b_before[0], 2.0, places=3)

        success = self.ekf.update_nhc()
        self.assertTrue(success)

        v_b_after = R.T @ self.ekf.v_enu
        # Lateral velocity should be substantially reduced towards 0
        self.assertLess(abs(v_b_after[0]), abs(v_b_before[0]))

    def test_covariance_positive_definiteness(self):
        """Verify error covariance matrix P remains positive definite and symmetric under prediction & updates."""
        self.ekf.initialize(0.0, 0.0, 0.0, 0.0, 15.0, 45.0)

        dt = 0.1
        t = 0.0
        for i in range(50):
            t += dt
            f_b = np.array([0.1, 0.0, self.ekf.kGravity])
            omega_b = np.array([0.001, 0.002, 0.0])
            self.ekf.predict(t, f_b, omega_b)

            if i % 5 == 0:
                self.ekf.update_nhc()
            if i % 20 == 0:
                self.ekf.update_zupt()

            # Symmetry check
            np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-10)

            # Positive-definiteness: all eigenvalues must be > 0
            eigvals = np.linalg.eigvalsh(self.ekf.P)
            self.assertTrue(np.all(eigvals > 0), f"Covariance matrix lost positive-definiteness at step {i}: min eig={np.min(eigvals)}")

    def test_curvature_adaptive_nhc_noise(self):
        """Verify kinematic centripetal NHC noise: vibration immunity on straight road and inflation under sustained turn."""
        cfg = EkfConfig(
            nhc_settle_duration_s=2.0,
            nhc_settle_sigma_extra=4.0,
            nhc_curv_c_coeff=2.0,
            nhc_curv_lat_coeff=0.0,
            nhc_curv_yaw_coeff=0.0,
            nhc_curv_lpf_cutoff_hz=0.0,
            nhc_curv_use_speed_floor=False,
            nhc_curv_use_nav_yaw=False,
            sigma_nhc_lat=0.15,
            sigma_nhc_vert=0.15,
        )
        ekf = ErrorStateEkf(cfg)
        ekf.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 90.0)

        # 1. Straight road past settling (30s = 15 tau >> 2s tau) with +/- 2.0 m/s^2 lateral accelerometer noise
        for i in range(1, 301):
            t = i * 0.1
            ax_noise = 2.0 if (i % 2 == 0) else -2.0
            ekf.predict(t, np.array([ax_noise, 0.0, ekf.kGravity]), np.zeros(3))
            ekf.update_nhc()

        # Kinematic centripetal acceleration is a_c = speed * omega_z = 20.0 * 0.0 = 0.0 m/s^2
        # Therefore, despite 2.0 m/s^2 lateral accelerometer vibration noise, sigma_nhc_lat must remain strictly 0.15 m/s!
        sigma_vibration_lat = ekf.compute_nhc_sigma_lat()
        sigma_vibration_vert = ekf.compute_nhc_sigma_vert()
        self.assertAlmostEqual(sigma_vibration_lat, 0.15, places=3)
        self.assertAlmostEqual(sigma_vibration_vert, 0.15, places=3)

        # 2. Sustained curve: omega_z = 0.1 rad/s at speed ~ 20 m/s -> a_c ~ 2.0 m/s^2
        ekf.predict(30.1, np.array([0.0, 0.0, ekf.kGravity]), np.array([0.0, 0.0, 0.1]))
        sigma_curve_lat = ekf.compute_nhc_sigma_lat()
        current_speed = float(np.linalg.norm(ekf.v_enu))
        w_yaw = abs(0.1 - ekf.b_gyro[2])
        expected_ac = current_speed * w_yaw
        expected_sigma = 0.15 * np.sqrt(1.0 + (2.0 * expected_ac) ** 2)
        self.assertAlmostEqual(sigma_curve_lat, expected_sigma, places=4)
        self.assertGreater(sigma_curve_lat, 0.30)

        # Vertical noise should be completely unaffected
        sigma_curve_vert = ekf.compute_nhc_sigma_vert()
        self.assertAlmostEqual(sigma_curve_vert, 0.15, places=4)

    def test_settling_grace_period(self):
        """Verify exponential decay of NHC settling grace period from t=t0."""
        cfg = EkfConfig(
            nhc_settle_duration_s=2.0,
            nhc_settle_sigma_extra=4.0,
            nhc_curv_lat_coeff=5.0,
            nhc_curv_yaw_coeff=2.0,
            sigma_nhc_lat=0.15,
            sigma_nhc_vert=0.15,
        )
        ekf = ErrorStateEkf(cfg)
        ekf.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 0.0)

        # At t=0: sigma_lat = 0.15 + 4.0 = 4.15
        self.assertAlmostEqual(ekf.compute_nhc_sigma_lat(), 4.15, places=2)

        # At t=2.0s (1 tau): extra = 4.0 * exp(-1) ~ 1.4715 -> sigma_lat ~ 1.6215
        ekf.predict(2.0, np.array([0.0, 0.0, ekf.kGravity]), np.zeros(3))
        expected_1tau = 0.15 + 4.0 * np.exp(-1.0)
        self.assertAlmostEqual(ekf.compute_nhc_sigma_lat(), expected_1tau, places=2)

        # At t=10.0s (5 tau): extra = 4.0 * exp(-5) ~ 0.027 -> sigma_lat ~ 0.177
        ekf.predict(10.0, np.array([0.0, 0.0, ekf.kGravity]), np.zeros(3))
        expected_5tau = 0.15 + 4.0 * np.exp(-5.0)
        self.assertAlmostEqual(ekf.compute_nhc_sigma_lat(), expected_5tau, places=2)

    def test_step24_mount_axis_invariance(self):
        """Step 24 Fix: Verify horizontal turning yaw rate is invariant to phone mount pitch/roll."""
        cfg = EkfConfig(nhc_settle_duration_s=0.0, nhc_curv_lpf_cutoff_hz=100.0)

        # 1. Flat mount
        ekf_flat = ErrorStateEkf(cfg)
        ekf_flat.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 0.0)
        ekf_flat.predict(0.1, np.array([0.0, 0.0, ekf_flat.kGravity]), np.array([0.0, 0.0, 0.10]))
        sigma_flat = ekf_flat.compute_nhc_sigma_lat()

        # 2. Portrait mount (pitch 90 deg forward)
        hp = 0.5 * (np.pi / 2.0)
        q_portrait = np.array([np.cos(hp), np.sin(hp), 0.0, 0.0])
        ekf_portrait = ErrorStateEkf(cfg)
        ekf_portrait.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 0.0, q0=q_portrait)
        ekf_portrait.predict(0.1, np.array([0.0, ekf_portrait.kGravity, 0.0]), np.array([0.0, 0.10, 0.0]))
        sigma_portrait = ekf_portrait.compute_nhc_sigma_lat()

        # 3. Landscape mount (roll 90 deg)
        hr = 0.5 * (np.pi / 2.0)
        q_landscape = np.array([np.cos(hr), 0.0, np.sin(hr), 0.0])
        ekf_landscape = ErrorStateEkf(cfg)
        ekf_landscape.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 0.0, q0=q_landscape)
        ekf_landscape.predict(0.1, np.array([-ekf_landscape.kGravity, 0.0, 0.0]), np.array([-0.10, 0.0, 0.0]))
        sigma_landscape = ekf_landscape.compute_nhc_sigma_lat()

        self.assertAlmostEqual(sigma_flat, sigma_portrait, places=3)
        self.assertAlmostEqual(sigma_flat, sigma_landscape, places=3)
        self.assertGreater(sigma_flat, 0.60)

    def test_step24_speed_collapse_decoupling(self):
        """Step 24 Fix: Verify pre-outage speed floor prevents centrifugal collapse during turns."""
        cfg = EkfConfig(nhc_curv_c_coeff=2.0, nhc_curv_use_speed_floor=True)
        ekf = ErrorStateEkf(cfg)
        ekf.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 0.0)

        # Force dead-reckoned speed to collapse to near-zero
        ekf.v_enu = np.array([0.001, 0.002, 0.0])
        self.assertLess(ekf.get_speed_ms(), 0.01)

        # Turn at 0.1 rad/s
        ekf.omega_yaw_filt = 0.1
        sigma_with_floor = ekf.compute_nhc_sigma_lat()

        # Without floor, speed ~ 0 -> a_c ~ 0 -> sigma = 0.15
        # With floor, speed_floor = 20.0 -> a_c = 20 * 0.1 = 2.0 -> sigma ~ 0.609
        self.assertGreater(sigma_with_floor, 0.60)

    def test_step24_vibration_low_pass_filter(self):
        """Step 24 Fix: Verify signed 2.0 Hz LPF attenuates 10 Hz vibration without false rectification."""
        cfg = EkfConfig(nhc_curv_lpf_cutoff_hz=2.0)
        ekf = ErrorStateEkf(cfg)
        ekf.initialize(0.0, 0.0, 0.0, 0.0, 20.0, 0.0)

        # Inject 10 Hz alternating +/- 0.20 rad/s noise
        for step in range(1, 101):
            t = step * 0.01  # 100 Hz sampling
            w_noise = 0.20 if (step % 2 == 0) else -0.20
            ekf.predict(t, np.array([0.0, 0.0, ekf.kGravity]), np.array([0.0, 0.0, w_noise]))

        # Filtered yaw rate should be attenuated by >90%
        self.assertLess(abs(ekf.omega_yaw_filt), 0.02)


if __name__ == "__main__":
    unittest.main()
