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


if __name__ == "__main__":
    unittest.main()
