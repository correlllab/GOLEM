"""Test livox_attitude: levelling FAST-LIO's camera_init, gyro propagation, units and the IMU-frame mount.

    python -m pytest test/test_livox_attitude.py
"""

import numpy as np
import pytest

from lmg_deploy import rotations as R
from lmg_deploy.livox_attitude import LivoxAttitude, in_imu_frame

DT = 0.005                          # s, the MID360's 200 Hz
LIVOX_PITCH = 0.1650   # rad, about the chip's 9.45 deg pitch on torso_link


def _rest(tilt_rotvec):
    """Return a still body's level-world orientation (tilt, a rotation vector) and its accelerometer reading in g."""
    q_wb = R.quat_from_rotvec(np.asarray(tilt_rotvec, float))
    return q_wb, R.quat_apply_inverse(q_wb, np.array([0.0, 0.0, 1.0]))


def _feed(att, t0, n, gyro, acc_g):
    """Feed n identical IMU samples from t0 (s); return the next stamp."""
    for k in range(n):
        att.add_imu(t0 + k * DT, gyro, acc_g)
    return t0 + n * DT


def test_levels_a_tilted_camera_init():
    """With camera_init at the tilted head's frame, calibrate() recovers the true projected gravity."""
    q_wb, acc_g = _rest([0.0, LIVOX_PITCH, 0.0])
    att = LivoxAttitude(acc_scale=9.81)
    _feed(att, 0.0, 250, np.zeros(3), acc_g)
    assert att.calibrate() == "no FAST-LIO pose yet"
    att.add_pose(1.0, [1.0, 0.0, 0.0, 0.0])              # camera_init is the body's frame at FAST-LIO's start
    assert att.calibrate() is None
    np.testing.assert_allclose(R.projected_gravity(att.quat()), R.projected_gravity(q_wb), atol=1e-9)


def test_catches_a_late_pose_up_with_the_gyro():
    """A pose that arrives late is carried by the gyro from its stamp to the latest sample."""
    att = LivoxAttitude(acc_scale=9.81)
    _, acc_g = _rest([0.0, 0.0, 0.0])
    t = _feed(att, 0.0, 250, np.zeros(3), acc_g)          # still and level: calibrate
    att.add_pose(t - DT, [1.0, 0.0, 0.0, 0.0])
    assert att.calibrate() is None
    w = np.array([0.5, 0.0, 0.0])                         # then roll at 0.5 rad/s for 1 s
    _feed(att, t, 201, w, acc_g)
    # FAST-LIO's pose for t + 0.6 arrives now, 0.4 s late, true to its stamp.
    att.add_pose(t + 0.6, R.quat_from_rotvec(w * 0.6))
    np.testing.assert_allclose(R.axis_angle_from_quat(att.quat()), w * 1.0, atol=1e-6)


def test_waits_for_stillness_and_g():
    """calibrate() refuses while the body turns or the accelerometer does not read g."""
    _, acc_g = _rest([0.1, 0.0, 0.0])
    att = LivoxAttitude(acc_scale=9.81)
    _feed(att, 0.0, 250, np.array([0.0, 0.0, 0.3]), acc_g)    # turning
    att.add_pose(1.0, [1.0, 0.0, 0.0, 0.0])
    assert "not still" in att.calibrate()
    att = LivoxAttitude(acc_scale=1.0)                    # g taken as m/s^2: |acc| 1
    _feed(att, 0.0, 250, np.zeros(3), acc_g)
    att.add_pose(1.0, [1.0, 0.0, 0.0, 0.0])
    assert "not g" in att.calibrate()
    with pytest.raises(RuntimeError):
        att.quat()


@pytest.mark.parametrize("unit, scale", [("g", 1.0), ("m/s^2", 9.81)])
def test_decides_the_accelerometer_unit(unit, scale):
    """With acc_scale None the unit is decided from the samples: g (real robot) or m/s^2 (simulators)."""
    q_wb, acc_g = _rest([0.0, LIVOX_PITCH, 0.0])
    att = LivoxAttitude(unit_samples=200)
    t = _feed(att, 0.0, 199, np.zeros(3), acc_g * scale)
    assert att.acc_scale is None
    _feed(att, t, 300, np.zeros(3), acc_g * scale)
    att.add_pose(2.0, [1.0, 0.0, 0.0, 0.0])
    assert att.calibrate() is None
    np.testing.assert_allclose(att.acc_scale, 9.81 if unit == "g" else 1.0)
    np.testing.assert_allclose(R.projected_gravity(att.quat()), R.projected_gravity(q_wb), atol=1e-9)


def test_in_imu_frame_rotates_gravity_by_the_mount():
    """The policy frame's projected gravity is the chip's rotated by r_imu_livox (livox -> policy frame)."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        q_l = R.quat_from_rotvec(rng.normal(0.0, 0.3, 3))
        q_mount = R.quat_from_rotvec(rng.normal(0.0, 0.2, 3))
        r = np.stack([R.quat_apply(q_mount, e) for e in np.eye(3)], axis=1)
        np.testing.assert_allclose(R.projected_gravity(in_imu_frame(q_l, r)), r @ R.projected_gravity(q_l), atol=1e-12)
    q = R.quat_from_rotvec(np.array([0.1, -0.2, 0.3]))
    np.testing.assert_allclose(in_imu_frame(q, np.eye(3)), R.quat_unique(q), atol=1e-12)
