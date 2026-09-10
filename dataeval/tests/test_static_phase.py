"""Unit and integration tests for Module A static-phase attitude initialization."""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from dataeval.harness.static_phase import (
    detect_stationary_window,
    estimate_gyro_bias,
    estimate_leveling,
    estimate_static_phase,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestStaticPhase(unittest.TestCase):
    """Test suite for static-phase attitude leveling and gyro bias estimation."""

    def setUp(self):
        self.g = 9.80665

    def _generate_synthetic_stationary_df(
        self,
        duration_s: float = 10.0,
        pitch_deg: float = 3.5,
        roll_deg: float = -2.0,
        gyro_bias_rad_s: Tuple[float, float, float] = (0.05, -0.02, 0.01),
        accel_noise_std: float = 0.0,
        gyro_noise_std: float = 0.0,
        speed_kmh: float = 0.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = int(duration_s * 10)  # 10 Hz
        t = np.linspace(0.0, duration_s - 0.1, n)

        pitch_rad = np.deg2rad(pitch_deg)
        roll_rad = np.deg2rad(roll_deg)

        # Theoretical specific force in body frame (X=Right, Y=Forward, Z=Up)
        # fx = -g * sin(roll) * cos(pitch)
        # fy =  g * sin(pitch)
        # fz =  g * cos(roll) * cos(pitch)
        fx_true = -self.g * np.sin(roll_rad) * np.cos(pitch_rad)
        fy_true = self.g * np.sin(pitch_rad)
        fz_true = self.g * np.cos(roll_rad) * np.cos(pitch_rad)

        ax = fx_true + rng.normal(0.0, accel_noise_std, n)
        ay = fy_true + rng.normal(0.0, accel_noise_std, n)
        az = fz_true + rng.normal(0.0, accel_noise_std, n)

        gx = gyro_bias_rad_s[0] + rng.normal(0.0, gyro_noise_std, n)
        gy = gyro_bias_rad_s[1] + rng.normal(0.0, gyro_noise_std, n)
        gz = gyro_bias_rad_s[2] + rng.normal(0.0, gyro_noise_std, n)

        return pd.DataFrame({
            "timestamp_s": t,
            "can_indicated_vehicle_speed_kmh": np.full(n, speed_kmh),
            "phone_gps_speed_kmh": np.full(n, speed_kmh),
            "phone_accel_x_m_s2": ax,
            "phone_accel_y_m_s2": ay,
            "phone_accel_z_m_s2": az,
            "phone_gyro_x_rad_s": gx,
            "phone_gyro_y_rad_s": gy,
            "phone_gyro_z_rad_s": gz,
        })

    def test_synthetic_clean_leveling_and_bias(self):
        """Verify exact mathematical recovery of known pitch, roll, and gyro bias on noiseless data."""
        true_pitch_deg = 3.5
        true_roll_deg = -2.0
        true_biases = (0.05, -0.02, 0.01)

        df = self._generate_synthetic_stationary_df(
            duration_s=10.0,
            pitch_deg=true_pitch_deg,
            roll_deg=true_roll_deg,
            gyro_bias_rad_s=true_biases,
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
        )

        res = estimate_static_phase(df, run_id="synth_clean")

        self.assertTrue(res.window_found)
        self.assertEqual(res.flag, "clean_initial")
        self.assertAlmostEqual(res.pitch_deg, true_pitch_deg, places=5)
        self.assertAlmostEqual(res.roll_deg, true_roll_deg, places=5)
        self.assertAlmostEqual(res.gyro_bias_x_rad_s, true_biases[0], places=6)
        self.assertAlmostEqual(res.gyro_bias_y_rad_s, true_biases[1], places=6)
        self.assertAlmostEqual(res.gyro_bias_z_rad_s, true_biases[2], places=6)
        self.assertAlmostEqual(res.accel_norm_m_s2, self.g, places=4)

    def test_synthetic_noisy_leveling_and_bias(self):
        """Verify noise suppression via temporal averaging on realistic sensor noise."""
        true_pitch_deg = 2.0
        true_roll_deg = -1.5
        true_biases = (-0.01, 0.03, -0.005)

        df = self._generate_synthetic_stationary_df(
            duration_s=15.0,
            pitch_deg=true_pitch_deg,
            roll_deg=true_roll_deg,
            gyro_bias_rad_s=true_biases,
            accel_noise_std=0.15,   # ~0.015g typical sensor noise
            gyro_noise_std=0.005,  # ~0.28 deg/s gyro noise
            seed=123,
        )

        res = estimate_static_phase(df, run_id="synth_noisy")

        self.assertTrue(res.window_found)
        # Recovered angles within 0.2 degree
        self.assertAlmostEqual(res.pitch_deg, true_pitch_deg, delta=0.2)
        self.assertAlmostEqual(res.roll_deg, true_roll_deg, delta=0.2)
        # Recovered biases within 0.001 rad/s (~0.05 deg/s)
        self.assertAlmostEqual(res.gyro_bias_x_rad_s, true_biases[0], delta=0.001)
        self.assertAlmostEqual(res.gyro_bias_y_rad_s, true_biases[1], delta=0.001)
        self.assertAlmostEqual(res.gyro_bias_z_rad_s, true_biases[2], delta=0.001)

    def test_no_window_found_path(self):
        """Verify that continuously moving data returns window_found=False and NaN values."""
        df = self._generate_synthetic_stationary_df(
            duration_s=20.0,
            speed_kmh=45.0,  # Driving at 45 km/h
            accel_noise_std=1.2,  # Road vibration
        )

        res = estimate_static_phase(df, run_id="synth_moving")

        self.assertFalse(res.window_found)
        self.assertEqual(res.flag, "no_window_found")
        self.assertEqual(res.duration_s, 0.0)
        self.assertTrue(np.isnan(res.pitch_rad))
        self.assertTrue(np.isnan(res.roll_rad))
        self.assertTrue(np.isnan(res.gyro_bias_x_rad_s))

    def test_parked_run_flagging(self):
        """Verify that fully parked runs (pair_Vw1, pair_Vw15) receive the parked_run flag."""
        df = self._generate_synthetic_stationary_df(duration_s=10.0)
        res = estimate_static_phase(df, run_id="pair_Vw1")

        self.assertTrue(res.window_found)
        self.assertEqual(res.flag, "parked_run")

    def test_real_paired_fixture(self):
        """Verify static phase execution on real checked-in fixture."""
        fixture_p = FIXTURES_DIR / "processed" / "paired" / "pair_Vfa01.parquet"
        self.assertTrue(fixture_p.exists(), f"Fixture missing: {fixture_p}")

        df = pd.read_parquet(fixture_p)
        res = estimate_static_phase(df, run_id="pair_Vfa01")

        self.assertIsInstance(res.window_found, (bool, np.bool_))
        if res.window_found:
            self.assertGreater(res.duration_s, 0.0)
            self.assertFalse(np.isnan(res.pitch_rad))
            self.assertFalse(np.isnan(res.roll_rad))


if __name__ == "__main__":
    unittest.main()
