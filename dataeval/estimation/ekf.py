"""15-State Error-State Extended Kalman Filter (ES-EKF) for Inertial Navigation.

Implements full strapdown navigation mechanics coupled with continuous
error-state estimation:
- State vector: delta_x = [delta_p (3), delta_v (3), delta_theta (3), b_gyro (3), b_accel (3)] in R^15.
- Zero-Velocity Updates (ZUPT) during stationary intervals.
- Non-Holonomic Constraints (NHC) during vehicle forward motion.
- Random walk time-varying estimation of 3D gyroscope and accelerometer biases.

Units:
- Position p: meters [m] in local East-North-Up (ENU) tangent plane.
- Velocity v: meters per second [m/s] in local ENU.
- Attitude error delta_theta: radians [rad] rotation vector in ENU.
- Gyroscope bias b_gyro: radians per second [rad/s] in body frame.
- Accelerometer bias b_accel: meters per second squared [m/s^2] in body frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Compute 3x3 skew-symmetric cross-product matrix [v]x."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [w, x, y, z] to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q1 * q2 for [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to active 3x3 rotation matrix R_b^n.
    
    Transforms vectors from body frame to navigation frame: v_nav = R * v_body.
    """
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=float)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v from body frame to navigation frame using quaternion q."""
    w, x, y, z = q
    qv = np.array([x, y, z], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def rot_vec_to_quat(rot_vec: np.ndarray) -> np.ndarray:
    """Convert rotation vector (angle * axis) to unit quaternion [w, x, y, z]."""
    theta = np.linalg.norm(rot_vec)
    if theta < 1e-12:
        return np.array([1.0, 0.5 * rot_vec[0], 0.5 * rot_vec[1], 0.5 * rot_vec[2]], dtype=float)
    half_theta = 0.5 * theta
    factor = np.sin(half_theta) / theta
    return np.array([
        np.cos(half_theta),
        rot_vec[0] * factor,
        rot_vec[1] * factor,
        rot_vec[2] * factor
    ], dtype=float)


@dataclass
class EkfConfig:
    """Tuning parameters and sensor noise specifications for 15-state ES-EKF.
    
    Noise parameters justified based on consumer smartphone MEMS IMU characteristics:
    - sigma_accel_noise: 0.10 m/s^2 (continuous velocity random walk ~ 0.03 m/s/sqrt(s))
    - sigma_gyro_noise:  0.01 rad/s (continuous angular random walk ~ 0.003 rad/sqrt(s))
    - sigma_accel_bias:  0.001 m/s^2/sqrt(s) (thermal random walk for accelerometer bias)
    - sigma_gyro_bias:   0.0001 rad/s/sqrt(s) (allows ~0.39 deg/s drift across 15 minutes)
    - sigma_zupt:        0.05 m/s (zero-velocity pseudo-measurement noise)
    - sigma_nhc_lat:     0.15 m/s (non-holonomic lateral velocity constraint noise)
    - sigma_nhc_vert:    0.15 m/s (non-holonomic vertical velocity constraint noise)
    """
    sigma_accel_noise: float = 0.10
    sigma_gyro_noise: float = 0.01
    sigma_accel_bias: float = 0.001
    sigma_gyro_bias: float = 0.0001

    sigma_zupt: float = 0.05
    sigma_nhc_lat: float = 0.15
    sigma_nhc_vert: float = 0.15

    # ZUPT detector thresholds
    zupt_gyro_thresh_rad_s: float = 0.05      # ~2.8 deg/s
    zupt_accel_std_thresh_m_s2: float = 0.25  # vibration threshold
    zupt_accel_mag_window_s: float = 0.6      # window duration for variance check
    zupt_speed_gate_m_s: float = 2.5          # velocity gate to prevent false trigger on highway


class ErrorStateEkf:
    """15-State Error-State Extended Kalman Filter for 3D INS with ZUPT and NHC."""

    def __init__(self, config: Optional[EkfConfig] = None):
        self.cfg = config or EkfConfig()
        self.kGravity = 9.80665
        self.kEarthRadius = 6371000.0

        # Nominal States
        self.t: float = 0.0
        self.p_enu: np.ndarray = np.zeros(3, dtype=float)     # Position [m] East, North, Up
        self.v_enu: np.ndarray = np.zeros(3, dtype=float)     # Velocity [m/s] East, North, Up
        self.q: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])   # Attitude [w, x, y, z] Body -> ENU
        self.b_gyro: np.ndarray = np.zeros(3, dtype=float)    # Gyro bias [rad/s] in Body
        self.b_accel: np.ndarray = np.zeros(3, dtype=float)   # Accel bias [m/s^2] in Body

        # Reference origin for WGS-84 conversion
        self.lat0: float = 0.0
        self.lon0: float = 0.0
        self.alt0: float = 0.0

        # Error Covariance Matrix P in R^{15 x 15}
        self.P: np.ndarray = np.eye(15, dtype=float)
        self._init_covariance()

        # Last kinematic acceleration for trapezoidal integration
        self.last_accel_nav: np.ndarray = np.zeros(3, dtype=float)

        # Buffers for stillness detection
        self.imu_buffer: List[Tuple[float, np.ndarray, np.ndarray]] = [] # (t, f_b, omega_b)
        self.is_initialized: bool = False

    def _init_covariance(self):
        """Initialize state error covariance matrix P with conservative initial uncertainties."""
        self.P = np.zeros((15, 15), dtype=float)
        # Position error covariance: 1.0 m^2 horizontal, 4.0 m^2 vertical
        self.P[0:3, 0:3] = np.diag([1.0, 1.0, 4.0])
        # Velocity error covariance: 0.25 (m/s)^2
        self.P[3:6, 3:6] = np.diag([0.25, 0.25, 0.25])
        # Attitude error covariance: (2 deg)^2 ~ 0.001 rad^2
        self.P[6:9, 6:9] = np.diag([0.001, 0.001, 0.002])
        # Gyro bias covariance: (0.5 deg/s)^2 ~ 7.6e-5 (rad/s)^2
        self.P[9:12, 9:12] = np.eye(3) * 1.0e-4
        # Accel bias covariance: (0.1 m/s^2)^2 ~ 0.01 (m/s^2)^2
        self.P[12:15, 12:15] = np.eye(3) * 0.01

    def initialize(
        self,
        t0: float,
        lat0: float,
        lon0: float,
        alt0: float,
        speed_ms: float,
        heading_deg: float,
        q0: Optional[Union[np.ndarray, Tuple[float, float, float, float]]] = None,
        b_gyro0: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
        b_accel0: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
        initial_accel: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize filter navigation state and reference origin."""
        self.t = t0
        self.lat0 = lat0
        self.lon0 = lon0
        self.alt0 = alt0

        self.p_enu = np.zeros(3, dtype=float)

        psi = np.radians(heading_deg)
        self.v_enu = np.array([speed_ms * np.sin(psi), speed_ms * np.cos(psi), 0.0], dtype=float)

        if q0 is not None:
            self.q = quat_normalize(np.array(q0, dtype=float))
        else:
            half_psi = 0.5 * psi
            self.q = np.array([np.cos(half_psi), 0.0, 0.0, -np.sin(half_psi)], dtype=float)

        self.b_gyro = np.array(b_gyro0, dtype=float) if b_gyro0 is not None else np.zeros(3, dtype=float)
        self.b_accel = np.array(b_accel0, dtype=float) if b_accel0 is not None else np.zeros(3, dtype=float)

        init_f = np.array(initial_accel, dtype=float) if initial_accel is not None else np.array([0.0, 0.0, self.kGravity])
        gravity_nav = np.array([0.0, 0.0, -self.kGravity], dtype=float)
        self.last_accel_nav = quat_rotate(self.q, init_f - self.b_accel) + gravity_nav

        self._init_covariance()
        self.imu_buffer = []
        self.is_initialized = True

    def predict(self, t: float, f_body: np.ndarray, omega_body: np.ndarray) -> None:
        """Propagate nominal state and error covariance forward over dt = t - self.t."""
        if not self.is_initialized:
            return

        dt = t - self.t
        if dt <= 0.0 or dt > 1.0:
            self.t = t
            return

        # Maintain buffer for stillness detection
        self.imu_buffer.append((t, f_body.copy(), omega_body.copy()))
        cutoff = t - self.cfg.zupt_accel_mag_window_s
        self.imu_buffer = [item for item in self.imu_buffer if item[0] >= cutoff]

        # 1. Unbiased sensor readings
        omega_unbiased = omega_body - self.b_gyro
        f_unbiased = f_body - self.b_accel

        # 2. Attitude propagation via quaternion rotation vector
        dq = rot_vec_to_quat(omega_unbiased * dt)
        q_prev = self.q.copy()
        self.q = quat_normalize(quat_multiply(self.q, dq))

        # 3. Specific force resolution and gravity subtraction
        R = quat_to_rot_matrix(self.q)
        f_nav = R @ f_unbiased
        accel_nav = f_nav + np.array([0.0, 0.0, -self.kGravity], dtype=float)

        # 4. Trapezoidal velocity and position propagation
        v_prev = self.v_enu.copy()
        delta_v = (self.last_accel_nav + accel_nav) * (0.5 * dt)
        self.v_enu += delta_v

        delta_p = (v_prev + self.v_enu) * (0.5 * dt)
        self.p_enu += delta_p

        self.last_accel_nav = accel_nav
        self.t = t

        # 5. Error state transition matrix Phi ~ I + F * dt
        F = np.zeros((15, 15), dtype=float)
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -skew_symmetric(f_nav)
        F[3:6, 12:15] = -R
        F[6:9, 9:12] = -R

        Phi = np.eye(15, dtype=float) + F * dt

        # 6. Discrete process noise matrix Qd
        Qd = np.zeros((15, 15), dtype=float)
        Qd[3:6, 3:6] = (self.cfg.sigma_accel_noise ** 2) * dt * np.eye(3)
        Qd[6:9, 6:9] = (self.cfg.sigma_gyro_noise ** 2) * dt * np.eye(3)
        Qd[9:12, 9:12] = (self.cfg.sigma_gyro_bias ** 2) * dt * np.eye(3)
        Qd[12:15, 12:15] = (self.cfg.sigma_accel_bias ** 2) * dt * np.eye(3)

        # Covariance time propagation: P = Phi * P * Phi^T + Qd
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

    def is_stationary(self) -> bool:
        """Detect whether vehicle is stationary using windowed accelerometer variance and gyro norm."""
        if len(self.imu_buffer) < 4:
            return False

        # Speed gate: if filter thinks we are going > 2.5 m/s, require very strong evidence
        current_speed = np.linalg.norm(self.v_enu)
        if current_speed > self.cfg.zupt_speed_gate_m_s:
            return False

        omegas = np.array([item[2] - self.b_gyro for item in self.imu_buffer])
        gyro_norms = np.linalg.norm(omegas, axis=1)
        if np.mean(gyro_norms) > self.cfg.zupt_gyro_thresh_rad_s:
            return False

        accels = np.array([item[1] for item in self.imu_buffer])
        accel_mags = np.linalg.norm(accels, axis=1)
        accel_std = np.std(accel_mags)
        accel_mean_dev = abs(np.mean(accel_mags) - self.kGravity)

        return (accel_std < self.cfg.zupt_accel_std_thresh_m_s2) and (accel_mean_dev < 0.6)

    def update_zupt(self) -> bool:
        """Apply Zero-Velocity Update (ZUPT) measurement."""
        # Measurement: z = 0 - v_enu = -v_enu
        z = -self.v_enu
        H = np.zeros((3, 15), dtype=float)
        H[0:3, 3:6] = np.eye(3)

        R_meas = (self.cfg.sigma_zupt ** 2) * np.eye(3)
        return self._apply_kalman_update(H, z, R_meas)

    def update_nhc(self) -> bool:
        """Apply Non-Holonomic Constraints (NHC) on lateral and vertical body velocity."""
        R = quat_to_rot_matrix(self.q)
        # Body velocity: v_b = R^T * v_enu
        # v_b[0] = lateral (Right), v_b[2] = vertical (Up)
        v_b = R.T @ self.v_enu

        # Residual: z = [0 - v_bx, 0 - v_bz] = [-v_bx, -v_bz]
        z = np.array([-v_b[0], -v_b[2]], dtype=float)

        # H matrix: rows 0 and 2 of (R^T * delta_v - R^T * [v_enu]x * delta_theta)
        r1 = R[:, 0] # first column of R (first row of R^T)
        r3 = R[:, 2] # third column of R (third row of R^T)

        v_skew = skew_symmetric(self.v_enu)

        H = np.zeros((2, 15), dtype=float)
        H[0, 3:6] = r1
        H[0, 6:9] = - (r1 @ v_skew)
        H[1, 3:6] = r3
        H[1, 6:9] = - (r3 @ v_skew)

        R_meas = np.diag([self.cfg.sigma_nhc_lat ** 2, self.cfg.sigma_nhc_vert ** 2])
        return self._apply_kalman_update(H, z, R_meas)

    def _apply_kalman_update(self, H: np.ndarray, z: np.ndarray, R_meas: np.ndarray) -> bool:
        """Perform standard Kalman update using Joseph stabilized covariance form."""
        S = H @ self.P @ H.T + R_meas
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        dx = K @ z

        # Inject error states into nominal states
        self.p_enu += dx[0:3]
        self.v_enu += dx[3:6]

        dtheta = dx[6:9]
        dq = rot_vec_to_quat(dtheta)
        self.q = quat_normalize(quat_multiply(dq, self.q))

        self.b_gyro += dx[9:12]
        self.b_accel += dx[12:15]

        # Joseph form covariance update: P = (I - KH) * P * (I - KH)^T + K * R * K^T
        I_KH = np.eye(15, dtype=float) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def get_position_lat_lon(self) -> Tuple[float, float, float]:
        """Convert local ENU tangent plane position to geodetic WGS-84 (lat, lon, alt)."""
        rad_to_deg = 180.0 / np.pi
        delta_lat = (self.p_enu[1] / self.kEarthRadius) * rad_to_deg
        lat = self.lat0 + delta_lat

        cos_lat = np.cos(np.radians(self.lat0))
        if abs(cos_lat) < 1e-6:
            lon = self.lon0
        else:
            delta_lon = (self.p_enu[0] / (self.kEarthRadius * cos_lat)) * rad_to_deg
            lon = self.lon0 + delta_lon

        alt = self.alt0 + self.p_enu[2]
        return lat, lon, alt

    def get_speed_ms(self) -> float:
        """Scalar horizontal speed in m/s."""
        return float(np.sqrt(self.v_enu[0] ** 2 + self.v_enu[1] ** 2))

    def get_heading_deg(self) -> float:
        """Extract azimuth heading in degrees clockwise from North [0, 360)."""
        fwd_nav = quat_rotate(self.q, np.array([0.0, 1.0, 0.0], dtype=float))
        heading = np.degrees(np.arctan2(fwd_nav[0], fwd_nav[1]))
        if heading < 0.0:
            heading += 360.0
        return float(heading)

    def get_covariance_trace(self) -> float:
        """Return trace of error covariance matrix P."""
        return float(np.trace(self.P))

    def get_position_uncertainty_m(self) -> float:
        """Return 1-sigma horizontal position uncertainty in meters."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
