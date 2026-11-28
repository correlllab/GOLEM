"""Low-pass the policy's angular velocity as training did: a 2nd-order Butterworth, exact at any sample rate.

marl_deploy_node runs it on /lowstate's torso-IMU gyroscope, each sample's dt from LowState.tick;
export.yaml's imu.gyro_lowpass_hz is the cutoff the run trained with (Locomanipulation_game's
deploy_parity.py compares it with training's filter).
"""

import numpy as np

MAX_DT = 0.05            # s; a longer gap between samples (a dropout) restarts the filter


class GyroLowpass:
    """Low-pass each axis with a 2nd-order Butterworth, the policy's angular-velocity filter.

    The continuous filter y'' + 2 zeta w y' + w^2 y = w^2 u (w = 2 pi cutoff_hz,
    zeta = 1/sqrt(2)), state x = (y, y'), discretized exactly (zero-order hold on u)
    over each sample's own dt. It is therefore the same filter at /lowstate's 500 Hz
    and at training's physics rate. It starts, and restarts after a dropout, at
    x = (u, 0). Training uses the same definition; export.yaml's
    imu.gyro_lowpass_hz says which cutoff a run trained with.
    """

    ZETA = 1.0 / np.sqrt(2.0)

    def __init__(self, cutoff_hz: float):
        """Set the cutoff (Hz); raise ValueError unless it is positive."""
        if cutoff_hz <= 0.0:
            raise ValueError(f"cutoff_hz must be positive, not {cutoff_hz}")
        w = 2.0 * np.pi * float(cutoff_hz)
        self._a = np.array([[0.0, 1.0], [-w * w, -2.0 * self.ZETA * w]])
        self._b = np.array([0.0, w * w])
        self._sigma, self._wd, self._w2 = self.ZETA * w, w * np.sqrt(1.0 - self.ZETA ** 2), w * w
        self._x: np.ndarray | None = None   # (2, axes): y and y' per axis

    def reset(self, u: np.ndarray) -> np.ndarray:
        """Restart the filter at rest on input u and return u."""
        u = np.asarray(u, dtype=np.float64)
        self._x = np.stack([u, np.zeros_like(u)])
        return u.copy()

    def _discrete(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return exp(A dt) in closed form (underdamped) and the ZOH input matrix A^-1 (exp(A dt) - I) B."""
        e, c, sn = np.exp(-self._sigma * dt), np.cos(self._wd * dt), np.sin(self._wd * dt)
        k = self._sigma / self._wd
        ad = e * np.array([[c + k * sn, sn / self._wd], [-self._w2 * sn / self._wd, c - k * sn]])
        bd = np.linalg.solve(self._a, (ad - np.eye(2)) @ self._b)
        return ad, bd

    def __call__(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Return the filtered value after sample u, held over the dt (s) since the last one.

        A first sample, or a dt outside (0, MAX_DT], restarts the filter at u.
        """
        if self._x is None or not 0.0 < dt <= MAX_DT:
            return self.reset(u)
        ad, bd = self._discrete(dt)
        self._x = ad @ self._x + np.outer(bd, np.asarray(u, dtype=np.float64))
        return self._x[0].copy()
