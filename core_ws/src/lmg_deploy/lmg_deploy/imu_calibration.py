"""Calibrate the H1-2's torso IMU (/lowstate imu_state) orientation: the correction, the export contract, the estimate.

The calibration is the attitude the quaternion reads with imu_link level, as (roll, pitch, yaw) in deg:
reported = true * R_bias (body frame), R_bias = Rz(yaw) Ry(pitch) Rx(roll) (Isaac Lab's quat_from_euler_xyz,
rotations.imu_mount_quat), pitch positive nose down, roll positive tilting right. Training draws its quaternion bias
the same way (Locomanipulation_game's lowstate_imu.py, LowstateImu.quaternion). The correction is
corrected = reported * R_bias^-1.

Training biases the quaternion only, so by default the gyroscope is left alone. With rotate_gyro (the offset is a
physical mounting tilt, so every vector the IMU measures is tilted too) the gyro is re-expressed in the level frame:
a frame B tilted from the torso frame T by R_bias (q_wB = q_wT R_bias) measures w_B = R_bias^-1 w_T, so
w_T = R_bias w_B (test_imu_calibration.py pins the direction on a rotating body).

export.yaml's imu.orientation says which IMU the policy trained on: "calibrated" (the bias removed, only a residual
of imu.residual_deg left) or "raw" (the robot's bias itself, imu.quat_bias_deg: every export before the key existed).
resolve_calibration() is the rule between that and the robot's calibration (config/imu_calibration.yaml).

The estimate (calibrate_imu.py): with both feet flat on level ground and the robot still, each sole is level up to
yaw, so forward kinematics sole -> imu_link gives imu_link's true tilt (fk_projected_gravity); the quaternion's tilt
against it is the bias. Yaw is unobservable this way and stays 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmg_deploy.rotations import imu_mount_quat, projected_gravity, quat_apply, quat_conjugate, quat_mul

ORIENTATIONS = ("raw", "calibrated")
CALIBRATION_LIMIT_DEG = 15.0   # a calibration angle beyond this is a typo, not an IMU offset
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")   # the soles: level at zero joint angles (URDF)
IMU_FRAME = "imu_link"
KNEES = (3, 9)                 # left_knee_joint, right_knee_joint in the policy's (Unitree motor) order
NUM_LEG_JOINTS = 12


# -- the correction -------------------------------------------------------------


def validate_calibration(calibration: dict | None) -> dict | None:
    """Return calibration with its defaults filled in (yaw_deg 0, rotate_gyro false), or None for None.

    Keys: roll_deg, pitch_deg (required), yaw_deg, rotate_gyro. Raises ValueError on a missing, unknown,
    non-finite or out-of-range (CALIBRATION_LIMIT_DEG) value.
    """
    if calibration is None:
        return None
    unknown = set(calibration) - {"roll_deg", "pitch_deg", "yaw_deg", "rotate_gyro"}
    if unknown:
        raise ValueError(f"IMU calibration has unknown keys {sorted(unknown)}")
    out = {"yaw_deg": 0.0, "rotate_gyro": False}
    out.update(calibration)
    for key in ("roll_deg", "pitch_deg", "yaw_deg"):
        if key not in calibration and key != "yaw_deg":
            raise ValueError(f"IMU calibration needs {key}")
        value = out[key]
        if value is None or not np.isfinite(float(value)) or abs(float(value)) > CALIBRATION_LIMIT_DEG:
            raise ValueError(f"IMU calibration {key} must be finite and within +-{CALIBRATION_LIMIT_DEG} deg, "
                             f"not {value!r}")
        out[key] = float(value)
    if not isinstance(out["rotate_gyro"], bool):
        raise ValueError(f"IMU calibration rotate_gyro must be true or false, not {out['rotate_gyro']!r}")
    return out


def bias_quat(calibration: dict) -> np.ndarray:
    """Return R_bias (wxyz), the attitude the quaternion reads with imu_link level: Rz(yaw) Ry(pitch) Rx(roll)."""
    return imu_mount_quat(*np.radians([calibration["roll_deg"], calibration["pitch_deg"],
                                       calibration.get("yaw_deg", 0.0)]))


def correct_imu(quat: np.ndarray, gyro: np.ndarray, calibration: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the calibrated (quaternion wxyz, gyro rad/s): quat * R_bias^-1, the gyro rotated by R_bias only with
    rotate_gyro. calibration None returns the inputs unchanged."""
    if calibration is None:
        return quat, gyro
    r = bias_quat(calibration)
    quat = quat_mul(np.asarray(quat, dtype=np.float64), quat_conjugate(r))
    if calibration.get("rotate_gyro", False):
        gyro = quat_apply(r, np.asarray(gyro, dtype=np.float64))
    return quat, gyro


def level_vector(v: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Return a vector the IMU measured (e.g. its specific force) in the level frame: rotated by R_bias with
    rotate_gyro (a physical tilt), else unchanged (an estimator offset tilts the quaternion only)."""
    if calibration is None or not calibration.get("rotate_gyro", False):
        return v
    return quat_apply(bias_quat(calibration), np.asarray(v, dtype=np.float64))


def load_calibration(path: str, robot: str) -> dict | None:
    """Return robot's entry of a calibration file (config/imu_calibration.yaml), validated; None if it is null or
    absent. Raises ValueError on a malformed entry, FileNotFoundError on a missing file."""
    import yaml
    with open(path) as f:
        table = yaml.safe_load(f) or {}
    if not isinstance(table, dict):
        raise ValueError(f"{path} must map robot names to calibrations")
    return validate_calibration(table.get(robot))


# -- the export contract ----------------------------------------------------------


def imu_orientation(cfg: dict) -> str:
    """Return export.yaml's imu.orientation, "raw" when absent (exports with quat_bias_deg, or no imu block).

    Raises ValueError on another value.
    """
    orientation = (cfg.get("imu") or {}).get("orientation", "raw")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"export.yaml imu.orientation must be one of {ORIENTATIONS}, not {orientation!r}: "
                         "this lmg_deploy doesn't know how that policy's IMU was trained")
    return orientation


def resolve_calibration(cfg: dict, calibration: dict | None) -> tuple[dict | None, str]:
    """Return the calibration to apply to the policy's IMU (None: none) and a line saying what was decided.

    imu.orientation calibrated: needs imu.source lowstate and a calibration (ValueError otherwise). raw (or
    absent): nothing is applied, even if a calibration is given: the policy trained on the biased IMU.
    """
    imu = cfg.get("imu") or {}
    orientation = imu_orientation(cfg)
    calibration = validate_calibration(calibration)
    if orientation == "raw":
        if calibration is None:
            return None, "IMU orientation raw (the policy trained on the biased IMU): no calibration applied"
        return None, ("IMU orientation raw (the policy trained on the biased IMU): the robot's IMU calibration "
                      f"(roll {calibration['roll_deg']:+.2f}, pitch {calibration['pitch_deg']:+.2f}, "
                      f"yaw {calibration['yaw_deg']:+.2f} deg) is ignored")
    if imu.get("source") != "lowstate":
        raise ValueError(f"export.yaml imu.orientation calibrated needs imu.source lowstate (the torso IMU the "
                         f"calibration is for), not {imu.get('source')!r}")
    if calibration is None:
        raise ValueError(
            "export.yaml imu.orientation is calibrated: the policy trained on an IMU with the robot's static offset "
            "removed, but this robot has no IMU calibration (config/imu_calibration.yaml, the robot's entry). "
            "Measure it (README 'IMU calibration', calibrate_imu) before running this policy")
    residual = imu.get("residual_deg")
    return calibration, (
        f"IMU orientation calibrated: applying roll {calibration['roll_deg']:+.2f}, pitch "
        f"{calibration['pitch_deg']:+.2f}, yaw {calibration['yaw_deg']:+.2f} deg, gyro "
        f"{'rotated' if calibration['rotate_gyro'] else 'as reported'}"
        + (f" (the policy tolerates a residual of {residual})" if residual else ""))


# -- the estimate -------------------------------------------------------------------


def fk_projected_gravity(q_pelvis_foot: np.ndarray, q_pelvis_imu: np.ndarray) -> np.ndarray:
    """Return imu_link's projected gravity if the foot (pelvis-frame quaternion) is level: its tilt by kinematics."""
    return projected_gravity(quat_mul(quat_conjugate(q_pelvis_foot), q_pelvis_imu))


def fk_imu_gravity(kin) -> np.ndarray:
    """Return (len(FEET), 3): imu_link's projected gravity by kinematics from each sole, taken level, at kin's last
    update() (kinematics.ArmKinematics with imu_link and FEET among its extra frames)."""
    _, q_imu = kin.frame_pose(IMU_FRAME)
    return np.array([fk_projected_gravity(kin.frame_pose(foot)[1], q_imu) for foot in FEET])


def tilt_deg(g: np.ndarray) -> np.ndarray:
    """Return the (roll, pitch) deg of a projected gravity (or an (N, 3) array of them): pitch positive nose down
    (gravity toward +x), roll positive tilting right (gravity toward -y)."""
    g = np.asarray(g, dtype=np.float64)
    return np.degrees(np.stack([np.arctan2(-g[..., 1], -g[..., 2]), np.arctan2(g[..., 0], -g[..., 2])], -1))


def estimate_bias(g_fk: np.ndarray, quats: np.ndarray, iters: int = 20) -> np.ndarray:
    """Return the (roll, pitch) deg of R_bias (yaw 0) that best carries the quaternions' tilts onto the kinematic ones.

    g_fk: (N, 3) imu_link's projected gravity by kinematics; quats: (N, 4) the reported wxyz quaternions. The
    corrected quaternion reads apply(R_bias, projected_gravity(q)); least squares over the samples, Gauss-Newton.
    """
    g_fk = np.atleast_2d(g_fk)
    g_rep = np.array([projected_gravity(q) for q in np.atleast_2d(quats)])
    x = np.zeros(2)

    def residual(x):
        r = imu_mount_quat(x[0], x[1], 0.0)
        return (np.array([quat_apply(r, g) for g in g_rep]) - g_fk).ravel()

    for _ in range(iters):
        f = residual(x)
        jac = np.stack([(residual(x + d) - f) / 1e-7 for d in np.eye(2) * 1e-7], 1)
        step = np.linalg.lstsq(jac, -f, rcond=None)[0]
        x += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return np.degrees(x)


@dataclass
class StillThresholds:
    """What counts as standing still on both feet, per /lowstate sample."""

    max_leg_dq: float = 0.2      # rad/s, every leg joint
    max_gyro: float = 0.1        # rad/s, the torso gyro's norm (unfiltered)
    knee_load_tau: float = 8.0   # Nm: each knee's tau_est at or below -this (extending against the body's weight;
                                 # 2026-11-20 bags: +6..7 hanging bent in the hoist, -14..-25 standing)


def still_and_loaded(dq: np.ndarray, gyro: np.ndarray, tau: np.ndarray, th: StillThresholds) -> np.ndarray:
    """Return, per sample, whether the legs and torso are still and both knees carry weight.

    dq, tau: (..., >=12) joint velocities and tau_est, policy order; gyro: (..., 3).
    """
    dq, gyro, tau = np.asarray(dq), np.asarray(gyro), np.asarray(tau)
    still = np.max(np.abs(dq[..., :NUM_LEG_JOINTS]), axis=-1) <= th.max_leg_dq
    still &= np.linalg.norm(gyro, axis=-1) <= th.max_gyro
    return still & np.all(tau[..., list(KNEES)] <= -th.knee_load_tau, axis=-1)


def still_windows(t: np.ndarray, mask: np.ndarray, min_s: float = 1.0, trim_s: float = 0.25) -> np.ndarray:
    """Return the samples inside runs of mask at least min_s long (t in s), trim_s cut from each run's ends."""
    keep = np.zeros(len(mask), dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        if t[stop - 1] - t[start] >= min_s:
            keep[start:stop] = (t[start:stop] >= t[start] + trim_s) & (t[start:stop] <= t[stop - 1] - trim_s)
    return keep


def calibrate(g_fk_left: np.ndarray, g_fk_right: np.ndarray, quats: np.ndarray) -> dict:
    """Return the calibration from still, loaded samples: each foot's (roll, pitch) estimate, their disagreement,
    and the result from both (their mean gravity), with the quaternion's and the kinematics' mean tilts."""
    left, right = estimate_bias(g_fk_left, quats), estimate_bias(g_fk_right, quats)
    g_both = np.atleast_2d(g_fk_left) + np.atleast_2d(g_fk_right)
    g_both /= np.linalg.norm(g_both, axis=1, keepdims=True)
    both = estimate_bias(g_both, quats)
    g_rep = np.array([projected_gravity(q) for q in np.atleast_2d(quats)])
    return {"roll_deg": float(both[0]), "pitch_deg": float(both[1]), "yaw_deg": 0.0,
            "left": left.tolist(), "right": right.tolist(), "feet_disagreement": np.abs(left - right).tolist(),
            "samples": int(len(np.atleast_2d(quats))),
            "fk_tilt": tilt_deg(g_both.mean(0)).tolist(), "imu_tilt": tilt_deg(g_rep.mean(0)).tolist()}


# -- the runtime check ----------------------------------------------------------------


class StandingTiltCheck:
    """Compare the policy's IMU tilt with the kinematic one once the robot has stood still for hold_s.

    update() once per control step; it returns a report (dict) once, the first time hold_s of consecutive standing
    has gathered, then None for good. The report's "mismatch" is true when the tilts differ by more than tol_deg.
    """

    def __init__(self, dt: float, hold_s: float = 5.0, tol_deg: float = 1.0):
        """Gather hold_s / dt consecutive standing steps."""
        self.need = max(1, int(round(hold_s / dt)))
        self.tol_deg = tol_deg
        self.done = False
        self._g_imu: list[np.ndarray] = []
        self._g_fk: list[np.ndarray] = []

    def update(self, standing: bool, g_imu: np.ndarray, g_fk: np.ndarray) -> dict | None:
        """Add one step: standing (still, both feet loaded, zero command), the IMU's projected gravity as the policy
        sees it and the kinematic one (both feet's mean); return the report once hold_s has gathered."""
        if self.done:
            return None
        if not standing:
            self._g_imu.clear()
            self._g_fk.clear()
            return None
        self._g_imu.append(np.asarray(g_imu, dtype=np.float64))
        self._g_fk.append(np.asarray(g_fk, dtype=np.float64))
        if len(self._g_imu) < self.need:
            return None
        self.done = True
        imu, fk = tilt_deg(np.mean(self._g_imu, 0)), tilt_deg(np.mean(self._g_fk, 0))
        diff = imu - fk
        return {"imu_tilt": imu.tolist(), "fk_tilt": fk.tolist(), "diff": diff.tolist(),
                "mismatch": bool(np.max(np.abs(diff)) > self.tol_deg)}
