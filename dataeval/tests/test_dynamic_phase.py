"""Unit and integration tests for Module B dynamic-phase yaw estimation and attitude synthesis."""

from pathlib import Path
from typing import Tuple
import unittest
import numpy as np
import pandas as pd

from dataeval.harness.dynamic_phase import (
    MEAN_EARTH_RADIUS_M,
    compute_heading_discrepancy,
    compute_pca_heading,
    construct_attitude_quaternion,
    detect_early_motion_window,
    estimate_dynamic_phase,
    integrate_gyro_heading_change,
    quat_multiply,
    quat_normalize,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDynamicPhase(unittest.TestCase):
    """Test suite for dynamic-phase PCA yaw estimation and quaternion synthesis."""

    def _generate_synthetic_straight_run(
        self,
        duration_s: float = 10.0,
        speed_ms: float = 15.0,
        heading_deg: float = 45.0,
        lat0: float = 52.0,
        lon0: float = -1.5,
    ) -> pd.DataFrame:
        n = int(duration_s * 10)
        t = np.linspace(0.0, duration_s - 0.1, n)
        psi_rad = np.deg2rad(heading_deg)

        # In ENU: East = sin(psi), North = cos(psi)
        v_e = speed_ms * np.sin(psi_rad)
        v_n = speed_ms * np.cos(psi_rad)

        e = v_e * t
        n_disp = v_n * t

        lat = lat0 + np.rad2deg(n_disp / MEAN_EARTH_RADIUS_M)
        lon = lon0 + np.rad2deg(e / (MEAN_EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))

        return pd.DataFrame({
            "timestamp_s": t,
            "can_indicated_vehicle_speed_kmh": np.full(n, speed_ms * 3.6),
            "phone_gps_speed_kmh": np.full(n, speed_ms * 3.6),
            "can_latitude_deg": lat,
            "can_longitude_deg": lon,
            "phone_latitude_deg": lat,
            "phone_longitude_deg": lon,
            "phone_gyro_z_rad_s": np.zeros(n),
        })

    def _generate_synthetic_turning_run(
        self,
        duration_s: float = 6.0,
        speed_ms: float = 12.0,
        initial_heading_deg: float = 0.0,
        yaw_rate_rad_s: float = 0.1,  # Turning left at ~5.73 deg/s
        lat0: float = 52.0,
        lon0: float = -1.5,
    ) -> pd.DataFrame:
        n = int(duration_s * 10)
        t = np.linspace(0.0, duration_s - 0.1, n)
        dt = 0.1

        # Turning left: heading psi decreases (d_psi/dt = -omega_z)
        psi_rad = np.deg2rad(initial_heading_deg) - yaw_rate_rad_s * t
        v_e = speed_ms * np.sin(psi_rad)
        v_n = speed_ms * np.cos(psi_rad)

        e = np.cumsum(v_e * dt)
        n_disp = np.cumsum(v_n * dt)

        lat = lat0 + np.rad2deg(n_disp / MEAN_EARTH_RADIUS_M)
        lon = lon0 + np.rad2deg(e / (MEAN_EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))

        return pd.DataFrame({
            "timestamp_s": t,
            "can_indicated_vehicle_speed_kmh": np.full(n, speed_ms * 3.6),
            "phone_gps_speed_kmh": np.full(n, speed_ms * 3.6),
            "can_latitude_deg": lat,
            "can_longitude_deg": lon,
            "phone_latitude_deg": lat,
            "phone_longitude_deg": lon,
            "phone_gyro_z_rad_s": np.full(n, yaw_rate_rad_s),
        })

    def test_synthetic_straight_trajectory_heading(self):
        """Verify PCA recovers known straight trajectory heading across quadrants."""
        for test_heading in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            df = self._generate_synthetic_straight_run(heading_deg=test_heading)
            h_rad, h_deg, dist = compute_pca_heading(df["can_latitude_deg"], df["can_longitude_deg"])
            self.assertGreater(dist, 100.0)
            self.assertAlmostEqual(h_deg, test_heading, delta=0.01)

    def test_synthetic_circular_arc_gyro_consistency(self):
        """Verify trajectory heading change and gyro-Z integrated heading change match on turning run."""
        yaw_rate = 0.1  # rad/s
        duration = 5.0  # s
        df = self._generate_synthetic_turning_run(duration_s=duration, yaw_rate_rad_s=yaw_rate)

        pca_delta, gyro_delta, disc = compute_heading_discrepancy(
            df["can_latitude_deg"],
            df["can_longitude_deg"],
            df["timestamp_s"],
            df["phone_gyro_z_rad_s"],
        )

        # Centroid time interval between the two halves is duration / 2 = 2.5s
        centroid_dt = duration / 2.0
        expected_heading_change = -np.rad2deg(yaw_rate * centroid_dt)

        # Both Gyro and PCA should match expected heading change
        self.assertAlmostEqual(gyro_delta, expected_heading_change, delta=0.5)
        self.assertAlmostEqual(pca_delta, expected_heading_change, delta=1.0)
        # Discrepancy should be small (< 1.0 degree)
        self.assertLess(abs(disc), 1.0)

    def test_no_motion_path(self):
        """Verify that parked runs with zero velocity return has_motion=False."""
        n = 50
        t = np.linspace(0.0, 4.9, n)
        df = pd.DataFrame({
            "timestamp_s": t,
            "can_indicated_vehicle_speed_kmh": np.zeros(n),
            "phone_gps_speed_kmh": np.zeros(n),
            "can_latitude_deg": np.full(n, 52.0),
            "can_longitude_deg": np.full(n, -1.5),
            "phone_gyro_z_rad_s": np.zeros(n),
        })

        res = estimate_dynamic_phase(df, run_id="synth_parked")
        self.assertFalse(res.has_motion)
        self.assertEqual(res.yaw_source, "none_parked")
        self.assertEqual(res.flag, "no_motion_found")
        self.assertTrue(np.isnan(res.yaw_rad))
        self.assertTrue(np.isnan(res.q_w))

    def test_attitude_quaternion_synthesis(self):
        """Verify quaternion synthesis matches analytical formula and normalizes to 1.0."""
        # Pure North heading (0 deg): q = [1, 0, 0, 0]
        q_north = construct_attitude_quaternion(yaw_rad=0.0, pitch_rad=0.0, roll_rad=0.0)
        self.assertAlmostEqual(q_north[0], 1.0, places=6)
        self.assertAlmostEqual(q_north[1], 0.0, places=6)
        self.assertAlmostEqual(q_north[2], 0.0, places=6)
        self.assertAlmostEqual(q_north[3], 0.0, places=6)

        # Pure East heading (90 deg = pi/2 rad): q = [cos(pi/4), 0, 0, -sin(pi/4)]
        q_east = construct_attitude_quaternion(yaw_rad=np.pi / 2.0, pitch_rad=0.0, roll_rad=0.0)
        self.assertAlmostEqual(q_east[0], np.sqrt(0.5), places=6)
        self.assertAlmostEqual(q_east[3], -np.sqrt(0.5), places=6)

        # Combined attitude
        yaw = np.deg2rad(60.0)
        pitch = np.deg2rad(5.0)
        roll = np.deg2rad(-3.0)
        qw, qx, qy, qz = construct_attitude_quaternion(yaw, pitch, roll)

        norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        self.assertAlmostEqual(norm, 1.0, places=7)

    def test_real_paired_fixture(self):
        """Verify dynamic phase estimation on checked-in fixture."""
        fixture_p = FIXTURES_DIR / "processed" / "paired" / "pair_Vfa01.parquet"
        self.assertTrue(fixture_p.exists(), f"Missing fixture: {fixture_p}")

        df = pd.read_parquet(fixture_p)
        res = estimate_dynamic_phase(df, run_id="pair_Vfa01")

        self.assertIsInstance(res.has_motion, (bool, np.bool_))
        if res.has_motion:
            self.assertFalse(np.isnan(res.yaw_deg))
            self.assertGreater(res.yaw_deg, 0.0)
            self.assertLess(res.yaw_deg, 360.0)
            self.assertFalse(np.isnan(res.q_w))


if __name__ == "__main__":
    unittest.main()
