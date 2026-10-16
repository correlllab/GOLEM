"""Unit tests for livox_imu.LivoxImu: units, mount rotation, attitude filter.

    python -m pytest test/test_livox_imu.py
The URDF mount test needs pinocchio and CL_Assets (H12_URDF overrides the path).
"""

import os

import numpy as np
import pytest

from lmg_deploy import rotations as R
from lmg_deploy.livox_imu import GRAVITY, LivoxImu

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.environ.get(
    "H12_URDF", os.path.join(HERE, "..", "..", "..", "..", "CL_Assets", "ros_assets", "h1_2_magpie.urdf"))
LIVOX_PITCH = 0.24015730507441985   # rad, livox_joint rpy in the URDF
DT = 0.005                          # s, the MID360's 200 Hz


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _feed(imu, r_world_imu, r_imu_livox, unit, n, t0=0.0, gyro_imu=np.zeros(3)):
    """n samples of a body at attitude r_world_imu, as the MID360 would read them."""
    up_imu = r_world_imu.T @ np.array([0.0, 0.0, 1.0])
    for k in range(n):
        imu.update(t0 + k * DT, r_imu_livox.T @ gyro_imu, r_imu_livox.T @ up_imu * unit)
    return t0 + n * DT


@pytest.mark.parametrize("unit", [1.0, GRAVITY])
def test_units_and_mount(unit):
    """Upright torso, tilted lidar: g or m/s^2 in, torso-frame gravity out."""
    r = _ry(LIVOX_PITCH)
    imu = LivoxImu(r, unit_samples=50)
    _feed(imu, np.eye(3), r, unit, 60)
    assert imu.ready
    assert imu.scale == pytest.approx(GRAVITY if unit == 1.0 else 1.0)
    np.testing.assert_allclose(imu.acc, [0.0, 0.0, GRAVITY], atol=1e-9)
    np.testing.assert_allclose(R.projected_gravity(imu.quat()), [0.0, 0.0, -1.0], atol=1e-9)


def test_static_tilt_matches_true_attitude():
    r = _ry(LIVOX_PITCH)
    att = _ry(0.06) @ _rx(-0.04)   # torso pitched forward 3.4 deg, rolled -2.3 deg
    imu = LivoxImu(r, unit_samples=10)
    _feed(imu, att, r, 1.0, 400)
    true = att.T @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(R.projected_gravity(imu.quat()), true, atol=1e-9)


def test_gyro_carries_attitude_through_a_bad_accelerometer():
    """A 0.2 rad/s pitch-up tracked by the gyro while the accelerometer still reads upright."""
    imu = LivoxImu(np.eye(3), tau_s=1e6, unit_samples=1)
    imu.update(0.0, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    imu.update(DT, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    rate = np.array([0.0, 0.2, 0.0])
    for k in range(2, 102):    # 0.5 s
        imu.update(k * DT, rate, np.array([0.0, 0.0, 1.0]))
    expect = _ry(0.2 * 100 * DT).T @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(R.projected_gravity(imu.quat()), expect, atol=1e-3)


def test_dropout_gap_not_integrated():
    imu = LivoxImu(np.eye(3), tau_s=1e6, unit_samples=1)
    imu.update(0.0, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    imu.update(1.0, np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(imu.up, [0.0, 0.0, 1.0], atol=1e-12)


def test_urdf_mount_is_the_livox_pitch():
    pytest.importorskip("pinocchio")
    if not os.path.isfile(URDF):
        pytest.skip(f"no URDF at {URDF}")
    from lmg_deploy.kinematics import ArmKinematics
    import yaml
    with open(os.path.join(HERE, "..", "policies", "explicit500", "explicit500.yaml")) as f:
        cfg = yaml.safe_load(f)
    names = cfg["joint_names"]
    kin = ArmKinematics(URDF, joint_names=names, ee_frames=cfg["arms"]["ee_frames"],
                        arm_joint_names=[names[13:20], names[20:27]],
                        extra_frames=[cfg["slam_frame"], "torso_link"])
    q = np.zeros(len(names))
    q[12] = 0.7   # torso yaw must not change the mount
    kin.update(q)
    q_rel = R.quat_mul(R.quat_conjugate(kin.frame_pose("torso_link")[1]), kin.frame_pose(cfg["slam_frame"])[1])
    r = np.stack([R.quat_apply(q_rel, e) for e in np.eye(3)], axis=1)
    np.testing.assert_allclose(r, _ry(LIVOX_PITCH), atol=1e-9)
