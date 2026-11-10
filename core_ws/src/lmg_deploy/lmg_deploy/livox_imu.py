"""Turn /livox/imu into the policy's IMU: gyro, specific force and attitude in the policy's IMU frame.

/livox/imu is taken to be in the URDF's livox_imu_link, the MID360's IMU chip,
upright. The H1-2's /livox/imu is the MID-360's raw, upside-down frame, which
this module does not handle: golem_bringup's lmg_real refuses to start for
that, and test_urdf_mount_is_the_livox_pitch is a strict xfail until it does.

The policy's IMU frame is the one it trained with (export.yaml ``imu.frame``):
livox_imu_link itself, livox_link (the point frame, 0.7 deg from the chip) or
the torso's imu_link for older runs. All sit on torso_link, so angular velocity
and gravity direction are the chip's rotated by the fixed mount (r_imu_livox,
from the URDF). Away from the chip the specific force also differs by the
lever-arm terms (alpha x r + omega x (omega x r), r about 0.4 m up the torso
for imu_link), which this leaves out: they vanish standing still and stay small
at walking rates.

/livox/imu carries no orientation (the driver leaves it zero), so LivoxImu's
attitude is a complementary filter on the up direction u (the unit specific
force at rest, IMU frame): the gyro carries it, du/dt = -omega x u, and each
sample pulls it toward the measured accelerometer direction with time constant
``tau_s``.

The MID360 reports acceleration in g, the simulators' /livox/imu in m/s^2. The
unit is decided once, from the mean magnitude of the first ``unit_samples``
samples, while the robot stands or hangs still before it engages.
"""

import numpy as np

from lmg_deploy.rotations import quat_apply, quat_conjugate, quat_from_matrix, quat_mul

GRAVITY = 9.81           # m/s^2
G_UNIT_MAX_NORM = 4.0    # a mean |acc| below this is in g (1.0), above it in m/s^2 (9.81)
MAX_DT = 0.05            # s; a longer gap between samples (a dropout) is not integrated

# The URDF frame /livox/imu is taken to be in: the MID360's IMU chip, upright.
LIVOX_IMU_FRAME = "livox_imu_link"


def livox_to_imu_rotation(kin, imu_frame: str, livox_frame: str = LIVOX_IMU_FRAME) -> np.ndarray:
    """Return the (3, 3) rotation taking livox_frame vectors into imu_frame, from the URDF.

    kin: a kinematics.ArmKinematics with both frames among its extra frames,
    updated at any joint state (both frames sit on torso_link, so the joints
    don't change the rotation).
    """
    q_rel = quat_mul(quat_conjugate(kin.frame_pose(imu_frame)[1]), kin.frame_pose(livox_frame)[1])
    return np.stack([quat_apply(q_rel, e) for e in np.eye(3)], axis=1)


class GyroLowpass:
    """Low-pass each axis with a 2nd-order Butterworth, the policy's angular-velocity filter.

    The continuous filter y'' + 2 zeta w y' + w^2 y = w^2 u (w = 2 pi cutoff_hz,
    zeta = 1/sqrt(2)), state x = (y, y'), discretized exactly (zero-order hold on u)
    over each sample's own dt. It is therefore the same filter at the MID360's 200 Hz
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


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the smallest rotation matrix taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        # Antiparallel: half a turn about any axis normal to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class LivoxImu:
    """Carry /livox/imu samples into the policy's IMU frame and filter an attitude from them."""

    def __init__(self, r_imu_livox: np.ndarray, tau_s: float = 0.5, unit_samples: int = 200,
                 gyro_lowpass_hz: float = 0.0):
        """Set the mount and filter constants; raise ValueError unless tau_s is positive.

        r_imu_livox: (3, 3) rotation taking /livox/imu vectors into the policy's IMU frame.
        tau_s: s, the attitude filter's time constant toward the accelerometer.
        unit_samples: samples that decide the accelerometer unit.
        gyro_lowpass_hz: cutoff of the GyroLowpass on ``gyro`` (the policy's angular
        velocity); 0 leaves it unfiltered. The attitude filter always integrates the
        raw rate.
        """
        if tau_s <= 0.0:
            raise ValueError(f"tau_s must be positive, not {tau_s}")
        self._r = np.asarray(r_imu_livox, dtype=np.float64)
        self._tau = float(tau_s)
        self._unit_samples = int(unit_samples)
        self._norms: list[float] = []
        self.scale: float | None = None     # m/s^2 per input unit, once decided
        self.gyro = np.zeros(3)             # rad/s, IMU frame, low-passed when gyro_lowpass_hz > 0
        self._gyro_lp = GyroLowpass(gyro_lowpass_hz) if gyro_lowpass_hz > 0.0 else None
        self.acc = np.zeros(3)              # m/s^2, IMU frame, +9.81 up at rest
        self._up: np.ndarray | None = None  # unit up direction, IMU frame
        self._stamp: float | None = None

    @property
    def ready(self) -> bool:
        """Return whether the attitude has its first sample."""
        return self._up is not None

    @property
    def up(self) -> np.ndarray:
        """Return the unit up direction (specific force at rest), IMU frame."""
        return self._up.copy()

    def update(self, stamp: float, gyro: np.ndarray, acc: np.ndarray) -> None:
        """Take one /livox/imu sample: stamp (s), gyro (rad/s) and acc (g or m/s^2), in livox_imu_link."""
        acc = np.asarray(acc, dtype=np.float64)
        if self.scale is None:
            self._norms.append(float(np.linalg.norm(acc)))
            if len(self._norms) < self._unit_samples:
                return
            self.scale = GRAVITY if np.mean(self._norms) < G_UNIT_MAX_NORM else 1.0
        rate = self._r @ np.asarray(gyro, dtype=np.float64)
        self.acc = self._r @ (acc * self.scale)
        dt = 0.0 if self._stamp is None else stamp - self._stamp
        self._stamp = stamp
        self.gyro = rate if self._gyro_lp is None else self._gyro_lp(rate, dt)
        norm = float(np.linalg.norm(self.acc))
        if norm < 1e-6:
            return
        meas_up = self.acc / norm
        if self._up is None:
            self._up = meas_up
            return
        if not 0.0 < dt <= MAX_DT:
            return
        up = self._up - dt * np.cross(rate, self._up)
        up += (dt / self._tau) * (meas_up - up)
        self._up = up / np.linalg.norm(up)

    def quat(self) -> np.ndarray:
        """Return the IMU attitude (wxyz, body -> world, zero yaw), whose projected_gravity() is -up."""
        return quat_from_matrix(_rotation_aligning(self._up, np.array([0.0, 0.0, 1.0])))
