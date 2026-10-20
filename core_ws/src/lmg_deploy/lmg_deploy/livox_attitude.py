"""The head's Livox MID-360 IMU attitude: gravity-aligned, from FAST-LIO, carried at the IMU rate by the gyro.

/livox/imu is in the URDF's livox_imu_link, the MID-360's IMU chip (upright, as livox_imu_upright publishes it), and
FAST-LIO's body frame is that IMU. The MID-360 has no orientation output, so the attitude comes from FAST-LIO:

  * /Odometry gives the IMU body's orientation in camera_init (the IMU frame at FAST-LIO's start, about 9.5 deg
    pitched on the H1-2's head, so not level), at the lidar rate and stamped at the scan's end;
  * between poses the orientation is carried forward by the gyro, q <- q exp(w dt), from the last pose's stamp, so
    the scan latency is made up and the attitude is current at the IMU rate (200 Hz);
  * calibrate() levels camera_init once: with the robot still, the accelerometer's mean is the up direction in the
    body, so q_align, the smallest rotation taking that direction (in camera_init) onto +z, makes q_align q a
    gravity-aligned attitude (its yaw is camera_init's, which the policy never uses).

The accelerometer arrives in g on the real robot (the Livox driver) and in m/s^2 from the simulators' bridges. With
acc_scale None the unit is decided once, as livox_imu.LivoxImu does, from the mean magnitude of the first
unit_samples samples (the robot holds the pre-pose then); acc_scale then turns it into the m/s^2 training saw. No ROS
here: marl_deploy_node feeds it.

On the real robot's bags (10-16, lmg_deploy/bags) this attitude and LivoxImu's complementary filter agree to 0.3 deg
standing, but differ by 1.9 deg on average (4.6 deg p95) while walking: the filter's pull toward the accelerometer
follows the gait's accelerations. The node falls back to the filter only while FAST-LIO is silent.

in_imu_frame() carries the attitude into the frame a policy trained with (export.yaml imu.frame: livox_imu_link
itself, livox_link, or the torso's imu_link), through the URDF mount livox_imu.livox_to_imu_rotation reads.
"""

from collections import deque

import numpy as np

from lmg_deploy.livox_imu import G_UNIT_MAX_NORM, GRAVITY
from lmg_deploy.rotations import quat_apply, quat_from_matrix, quat_from_rotvec, quat_mul, quat_unique

Z = np.array([0.0, 0.0, 1.0])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation (wxyz) taking unit vector a onto unit vector b."""
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        ortho = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        return quat_from_rotvec(np.pi * ortho / np.linalg.norm(ortho))
    return quat_from_rotvec(axis / s * np.arctan2(s, c))


def in_imu_frame(q_livox_w: np.ndarray, r_imu_livox: np.ndarray) -> np.ndarray:
    """The attitude (wxyz, body -> world) of the policy's IMU frame from /livox/imu's: r_imu_livox takes livox_imu_link
    vectors into that frame (livox_imu.livox_to_imu_rotation), so the frame's axes in livox_imu_link are its rows."""
    return quat_unique(quat_mul(q_livox_w, quat_from_matrix(np.asarray(r_imu_livox, dtype=np.float64).T)))


class LivoxAttitude:
    def __init__(self, acc_scale: float | None = None, buffer_s: float = 2.0, still_window_s: float = 1.0,
                 still_gyro: float = 0.1, still_acc_spread: float = 0.1, unit_samples: int = 200):
        """acc_scale: m/s^2 per accelerometer unit (9.81 for g), or None to decide it from the first unit_samples
        samples. still_gyro (rad/s) and still_acc_spread (the largest deviation from the window's mean accelerometer,
        as a share of its norm: h12_slam level_odom's max_accel_spread) bound how still the robot must be for
        calibrate()."""
        self.acc_scale = None if acc_scale is None else float(acc_scale)
        self._unit_samples = int(unit_samples)
        self._norms: list[float] = []
        self.buffer_s, self.still_window_s = float(buffer_s), float(still_window_s)
        self.still_gyro, self.still_acc_spread = float(still_gyro), float(still_acc_spread)
        self._imu: deque = deque()                 # (t, gyro rad/s, acc m/s^2), oldest first
        self._pose: tuple[float, np.ndarray] | None = None   # (stamp, camera_init <- body, wxyz)
        self._q: np.ndarray | None = None          # camera_init <- body now
        self._t: float | None = None               # the time _q is at
        self._align: np.ndarray | None = None      # level <- camera_init

    # -- inputs ------------------------------------------------------------------------
    def add_imu(self, t: float, gyro, acc) -> None:
        """One /livox/imu sample: stamp (s), angular velocity (rad/s), linear acceleration (acc_scale units)."""
        gyro = np.asarray(gyro, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        if self.acc_scale is None:
            self._norms.append(float(np.linalg.norm(acc)))
            if len(self._norms) < self._unit_samples:
                return
            self.acc_scale = GRAVITY if np.mean(self._norms) < G_UNIT_MAX_NORM else 1.0
        acc = acc * self.acc_scale
        if self._imu and t <= self._imu[-1][0]:
            return                                 # out of order or repeated
        if self._q is not None and self._imu and t > self._t:
            # the previous sample's rate held from where _q is to now
            self._q = quat_mul(self._q, quat_from_rotvec(self._imu[-1][1] * (t - self._t)))
            self._q /= np.linalg.norm(self._q)
            self._t = t
        self._imu.append((float(t), gyro, acc))
        while self._imu and self._imu[0][0] < t - self.buffer_s:
            self._imu.popleft()

    def add_pose(self, t: float, quat_wxyz) -> None:
        """One FAST-LIO pose: stamp (s, the scan's end) and the body's orientation in camera_init (wxyz)."""
        q = np.asarray(quat_wxyz, dtype=np.float64)
        q = quat_unique(q / np.linalg.norm(q))
        self._pose = (float(t), q)
        latest = self._imu[-1][0] if self._imu else t
        self._q, self._t = (self._integrate(q, t, latest), latest) if latest > t else (q, t)

    # -- calibration ----------------------------------------------------------------------
    def calibrate(self) -> str | None:
        """Level camera_init from the last still_window_s of samples: None when done, else why it can't yet."""
        if self._q is None:
            return "no FAST-LIO pose yet"
        if not self._imu:
            return "no /livox/imu yet"
        t_end = self._imu[-1][0]
        window = [s for s in self._imu if s[0] >= t_end - self.still_window_s]
        if len(window) < 10 or t_end - window[0][0] < 0.8 * self.still_window_s:
            return f"only {len(window)} IMU samples over the last {self.still_window_s:.1f} s"
        gyro = np.stack([s[1] for s in window])
        acc = np.stack([s[2] for s in window])
        if np.linalg.norm(gyro, axis=1).max() > self.still_gyro:
            return f"not still: gyro up to {np.linalg.norm(gyro, axis=1).max():.3f} rad/s"
        mean = acc.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if not 7.0 < norm < 12.5:
            return f"|acc| {norm:.2f} m/s^2 after acc_scale {self.acc_scale}: not g"
        spread = float(np.linalg.norm(acc - mean, axis=1).max()) / norm
        if spread > self.still_acc_spread:
            return f"not still: accelerometer spread {spread:.2f} of |g| (limit {self.still_acc_spread})"
        up_body = mean / np.linalg.norm(mean)     # at rest the accelerometer reads +g, up
        self._align = rotation_between(quat_apply(self._q, up_body), Z)
        return None

    @property
    def calibrated(self) -> bool:
        return self._align is not None

    # -- outputs --------------------------------------------------------------------------
    def latest_imu_time(self) -> float | None:
        return self._imu[-1][0] if self._imu else None

    def pose_time(self) -> float | None:
        return self._pose[0] if self._pose else None

    def quat(self) -> np.ndarray:
        """The body's gravity-aligned orientation (wxyz): projected gravity and tilt from it are the training env's."""
        if self._align is None or self._q is None:
            raise RuntimeError("LivoxAttitude: not calibrated")
        return quat_unique(quat_mul(self._align, self._q))

    def rates(self) -> tuple[np.ndarray, np.ndarray]:
        """(gyro rad/s, acc m/s^2) of the latest sample, as the training env's IMU at the step's last substep."""
        _, gyro, acc = self._imu[-1]
        return gyro.copy(), acc.copy()

    # -- internals ----------------------------------------------------------------------
    def _integrate(self, q: np.ndarray, t0: float, t1: float) -> np.ndarray:
        """q carried from t0 to t1 by the buffered gyro (zero-order hold of each sample until the next)."""
        samples = list(self._imu)
        for i, (t, gyro, _) in enumerate(samples):
            t_next = samples[i + 1][0] if i + 1 < len(samples) else t1
            lo, hi = max(t, t0), min(t_next, t1)
            if hi > lo:
                q = quat_mul(q, quat_from_rotvec(gyro * (hi - lo)))
        return q / np.linalg.norm(q)
