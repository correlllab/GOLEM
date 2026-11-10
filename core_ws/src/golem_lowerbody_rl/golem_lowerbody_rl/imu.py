"""
Correct /lowstate's IMU orientation for its mounting and project gravity into its frame.

Quaternions are (w, x, y, z), the order of unitree_hg's IMUState.quaternion.
"""

import numpy as np

__all__ = ['apply_trim', 'gravity_z', 'projected_gravity', 'trim_quaternion']

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
# gravity's direction in the world frame
_DOWN = np.array([0.0, 0.0, -1.0])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return the Hamilton product q1 * q2 in float32."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def trim_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Return the correction quaternion (float32) for a fixed IMU mounting bias.

    The angles (deg) are what the raw IMU reads when the body is level: positive pitch
    reads tilted forward, positive roll tilted right. The bias is the body-frame mounting
    rotation Rz(yaw) Ry(pitch) Rx(roll); apply_trim right-multiplies a reading by its
    conjugate, which this returns, so the corrected orientation reads level. Zero angles
    give IDENTITY.
    """
    if roll_deg == 0.0 and pitch_deg == 0.0 and yaw_deg == 0.0:
        return IDENTITY.copy()
    roll, pitch, yaw = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    hr, hp, hy = 0.5 * roll, 0.5 * pitch, 0.5 * yaw
    qx = np.array([np.cos(hr), np.sin(hr), 0.0, 0.0], dtype=np.float32)
    qy = np.array([np.cos(hp), 0.0, np.sin(hp), 0.0], dtype=np.float32)
    qz = np.array([np.cos(hy), 0.0, 0.0, np.sin(hy)], dtype=np.float32)
    q_off = _quat_mul(_quat_mul(qz, qy), qx)
    return np.array([q_off[0], -q_off[1], -q_off[2], -q_off[3]], dtype=np.float32)


def apply_trim(q_wxyz: np.ndarray, trim: np.ndarray) -> np.ndarray:
    """
    Return the reading q_wxyz corrected by trim, a trim_quaternion.

    The identity trim returns the reading's own values and dtype; any other returns the
    float32 product q_wxyz * trim.
    """
    if np.array_equal(trim, IDENTITY):
        return np.array(q_wxyz)
    return _quat_mul(q_wxyz, trim)


def projected_gravity(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Return gravity's unit direction in the IMU frame, (0, 0, -1) when level.

    The quaternion is normalised first. Raises ValueError for a quaternion that is not
    finite or has zero norm, which is no rotation.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f'the IMU quaternion {q.tolist()} is not a rotation')
    w, u = q[0] / norm, q[1:] / norm
    # _DOWN rotated by the inverse of q
    return (_DOWN * (2.0 * w * w - 1.0) - np.cross(u, _DOWN) * w * 2.0
            + u * np.dot(u, _DOWN) * 2.0)


def gravity_z(q_wxyz: np.ndarray) -> float:
    """Return gravity's z in the IMU frame: -1 upright, 0 lying down, 1 upside down."""
    return float(projected_gravity(q_wxyz)[2])
