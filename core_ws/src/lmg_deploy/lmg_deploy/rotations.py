"""Quaternion and frame helpers, numpy, single poses.

Quaternions are (w, x, y, z), as in Isaac Lab and the Unitree IMU. Each
function mirrors the isaaclab.utils.math function of the same name, which
the training environment used, so the deployed observations and IK step
follow the same conventions.
"""

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_unique(q: np.ndarray) -> np.ndarray:
    """The same rotation with qw >= 0."""
    return -q if q[0] < 0 else q


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q."""
    xyz = q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[0] * t + np.cross(xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q^-1: a world vector expressed in the frame q."""
    return quat_apply(quat_conjugate(q), v)


def axis_angle_from_quat(q: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    q = -q if q[0] < 0 else q
    mag = np.linalg.norm(q[1:])
    half_angle = np.arctan2(mag, q[0])
    angle = 2.0 * half_angle
    ratio = np.sin(half_angle) / angle if abs(angle) > eps else 0.5 - angle * angle / 48.0
    return q[1:] / ratio


def quat_from_rotvec(rotvec: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Quaternion for a rotation of |rotvec| rad about rotvec (isaaclab quat_from_angle_axis)."""
    angle = np.linalg.norm(rotvec)
    axis = rotvec / max(angle, eps)
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion, qw >= 0."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return quat_unique(q)


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Unit gravity direction in the body frame: (0, 0, -1) when upright."""
    return quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0]))


def subtract_frame_transforms(pos_a, quat_a, pos_b, quat_b):
    """Pose b expressed in frame a: T_a^-1 o T_b."""
    quat_a_inv = quat_conjugate(quat_a)
    return quat_apply(quat_a_inv, pos_b - pos_a), quat_mul(quat_a_inv, quat_b)


def compute_pose_error(pos_cur, quat_cur, pos_des, quat_des):
    """(position error, axis-angle error) from the current pose to the desired one, as isaaclab computes them."""
    quat_err = quat_mul(quat_des, quat_conjugate(quat_cur) / np.dot(quat_cur, quat_cur))
    return pos_des - pos_cur, axis_angle_from_quat(quat_err)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


def align_tilt(delta_quat: np.ndarray, up_prev: np.ndarray, up: np.ndarray) -> np.ndarray:
    """delta_quat (previous pelvis frame <- new, wxyz) turned by the smallest rotation that makes it carry the world's
    vertical in the new frame, up, onto the one in the previous frame, up_prev: its roll and pitch then agree with
    gravity's and its yaw is kept (Locomanipulation_game odometry.align_tilt)."""
    carried = quat_apply(delta_quat, up)
    axis = np.cross(carried, up_prev)
    sin, cos = np.linalg.norm(axis), float(carried @ up_prev)
    axis = axis / max(sin, 1e-9)
    return quat_mul(quat_from_rotvec(axis * np.arctan2(sin, cos)), delta_quat)
