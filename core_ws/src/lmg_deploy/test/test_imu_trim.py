"""Test the IMU trim (imu_mount_quat) as marl_deploy_node applies it.

    python -m pytest test/test_imu_trim.py
"""

import numpy as np

from lmg_deploy import rotations as R

DOWN = np.array([0.0, 0.0, -1.0])


def _trim(mount, quat, gyro, acc):
    """Apply marl_deploy_node._robot_state's trim to an attitude, a rate and a specific force."""
    return (R.quat_mul(quat, R.quat_conjugate(mount)),
            R.quat_apply(mount, gyro), R.quat_apply(mount, acc))


def test_positive_pitch_reads_forward_and_positive_roll_right():
    """A positive pitch trim reads tilted forward, a positive roll trim tilted right."""
    # Projected gravity +x: gravity toward the front, the torso pitched forward.
    assert R.projected_gravity(R.imu_mount_quat(0.0, 0.05, 0.0))[0] > 0.0
    # +y is left, so gravity toward -y: rolled right.
    assert R.projected_gravity(R.imu_mount_quat(0.05, 0.0, 0.0))[1] < 0.0


def test_level_torso_reads_level_after_trim():
    """A level torso reads level, and +g up, once its mount is trimmed out."""
    mount = R.imu_mount_quat(np.radians(0.5), np.radians(-2.5), np.radians(1.0))
    # A level torso: the IMU reads its mount and measures +g up in its own frame.
    raw_acc = R.quat_apply_inverse(mount, np.array([0.0, 0.0, 9.81]))
    quat, _, acc = _trim(mount, mount, np.zeros(3), raw_acc)
    np.testing.assert_allclose(R.projected_gravity(quat), DOWN, atol=1e-12)
    np.testing.assert_allclose(acc, [0.0, 0.0, 9.81], atol=1e-12)


def test_trimmed_accelerometer_agrees_with_trimmed_gravity():
    """After the trim the accelerometer and projected gravity agree on the tilt."""
    rng = np.random.default_rng(0)
    mount = R.imu_mount_quat(*np.radians([2.0, -2.5, 0.0]))
    for _ in range(20):
        att = R.quat_from_rotvec(rng.normal(0.0, 0.2, 3))   # the IMU's raw attitude reading
        raw_acc = R.quat_apply_inverse(att, np.array([0.0, 0.0, 9.81]))
        quat, _, acc = _trim(mount, att, np.zeros(3), raw_acc)
        np.testing.assert_allclose(acc / 9.81, -R.projected_gravity(quat), atol=1e-12)


def test_matches_lowerbody_composition_order():
    """The trim composes Rz(yaw) Ry(pitch) Rx(roll), the bias golem_lowerbody_rl.imu.trim_quaternion uses."""
    r, p, y = 0.1, -0.2, 0.3
    half = lambda a, i: np.array([np.cos(a / 2), *(np.sin(a / 2) * np.eye(3)[i])])
    expect = R.quat_mul(R.quat_mul(half(y, 2), half(p, 1)), half(r, 0))
    np.testing.assert_allclose(R.imu_mount_quat(r, p, y), expect, atol=1e-12)
