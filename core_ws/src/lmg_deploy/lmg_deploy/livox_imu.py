"""The MID360's IMU as the policy's IMU: gyro, accelerometer and attitude in the policy's IMU frame.

/livox/imu is in the URDF's livox_imu_link, the MID360's IMU chip. The policy's IMU
frame is the one it trained with (export.yaml ``imu.frame``): livox_imu_link itself,
livox_link (the point frame, 0.7 deg from the chip) or the torso's imu_link for older
runs. All sit on torso_link, so angular velocity and gravity direction are the chip's
rotated by the fixed mount (r_imu_livox, from the URDF). Away from the chip the
specific force also differs by the lever-arm terms (alpha x r + omega x (omega x r),
r about 0.4 m up the torso for imu_link), which this leaves out: they vanish standing
still and stay small at walking rates.

/livox/imu (livox_imu_upright's output, livox_link frame) carries no orientation:
the driver leaves it zero. The attitude is a complementary filter on the up
direction u (the unit specific force at rest, torso frame): the gyro carries it,
du/dt = -omega x u, and each sample pulls it toward the measured accelerometer
direction with time constant ``tau_s``.

The MID360 reports acceleration in g, the simulators' /livox/imu in m/s^2. The unit
is decided once, from the mean magnitude of the first ``unit_samples`` samples (the
robot is standing or hanging still before it engages).
"""

import numpy as np

from lmg_deploy.rotations import quat_apply, quat_conjugate, quat_from_matrix, quat_mul

GRAVITY = 9.81           # m/s^2
G_UNIT_MAX_NORM = 4.0    # a mean |acc| below this is in g (1.0), above it in m/s^2 (9.81)
MAX_DT = 0.05            # s; a longer gap between samples (a dropout) is not integrated


LIVOX_IMU_FRAME = "livox_imu_link"   # URDF frame of /livox/imu: the MID360's IMU chip, upright


def livox_to_imu_rotation(kin, imu_frame: str, livox_frame: str = LIVOX_IMU_FRAME) -> np.ndarray:
    """(3, 3) rotation taking livox_frame vectors into imu_frame, from the URDF.

    kin: a kinematics.ArmKinematics updated at any joint state that has both frames
    (they share torso_link, so the joints don't change it)."""
    q_rel = quat_mul(quat_conjugate(kin.frame_pose(imu_frame)[1]), kin.frame_pose(livox_frame)[1])
    return np.stack([quat_apply(q_rel, e) for e in np.eye(3)], axis=1)


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        # antiparallel: half a turn about any axis normal to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class LivoxImu:
    def __init__(self, r_imu_livox: np.ndarray, tau_s: float = 0.5, unit_samples: int = 200):
        """r_imu_livox: (3, 3) rotation taking livox_link vectors into the policy's IMU frame."""
        if tau_s <= 0.0:
            raise ValueError(f"tau_s must be positive, not {tau_s}")
        self._r = np.asarray(r_imu_livox, dtype=np.float64)
        self._tau = float(tau_s)
        self._unit_samples = int(unit_samples)
        self._norms: list[float] = []
        self.scale: float | None = None     # m/s^2 per input unit, once decided
        self.gyro = np.zeros(3)             # rad/s, IMU frame
        self.acc = np.zeros(3)              # m/s^2, IMU frame, +9.81 up at rest
        self._up: np.ndarray | None = None  # unit up direction, IMU frame
        self._stamp: float | None = None

    @property
    def ready(self) -> bool:
        return self._up is not None

    @property
    def up(self) -> np.ndarray:
        """Unit up direction (specific force at rest), IMU frame."""
        return self._up.copy()

    def update(self, stamp: float, gyro: np.ndarray, acc: np.ndarray) -> None:
        """One /livox/imu sample: stamp in s, gyro rad/s and acc (g or m/s^2) in livox_link."""
        acc = np.asarray(acc, dtype=np.float64)
        if self.scale is None:
            self._norms.append(float(np.linalg.norm(acc)))
            if len(self._norms) < self._unit_samples:
                return
            self.scale = GRAVITY if np.mean(self._norms) < G_UNIT_MAX_NORM else 1.0
        self.gyro = self._r @ np.asarray(gyro, dtype=np.float64)
        self.acc = self._r @ (acc * self.scale)
        norm = float(np.linalg.norm(self.acc))
        if norm < 1e-6:
            return
        meas_up = self.acc / norm
        dt = 0.0 if self._stamp is None else stamp - self._stamp
        self._stamp = stamp
        if self._up is None:
            self._up = meas_up
            return
        if not 0.0 < dt <= MAX_DT:
            return
        up = self._up - dt * np.cross(self.gyro, self._up)
        up += (dt / self._tau) * (meas_up - up)
        self._up = up / np.linalg.norm(up)

    def quat(self) -> np.ndarray:
        """IMU attitude, wxyz, zero yaw: projected_gravity() of it is -up."""
        # body -> world: the rotation taking the body's up direction onto world +z
        return quat_from_matrix(_rotation_aligning(self._up, np.array([0.0, 0.0, 1.0])))
