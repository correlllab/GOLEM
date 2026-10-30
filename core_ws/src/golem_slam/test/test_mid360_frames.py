"""
The URDF's MID360 frames as h12_navigation.launch.py reads them.

FAST-LIO's extrinsic (the lidar in the IMU frame) must stay near identity: both
streams arrive upright (MID360_config.json roll 180, livox_imu_upright). A 180 deg
rotation here would mean the URDF lost track of the inversion.
"""

import os

import numpy as np
import pytest

from golem_slam.robot_geometry import fixed_transform

URDF = os.environ.get('H12_URDF', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'CL_Assets', 'ros_assets',
    'h1_2_magpie_ros.urdf'))


@pytest.fixture(scope='module')
def urdf():
    if not os.path.isfile(URDF):
        pytest.skip(f'no URDF at {URDF}')
    with open(URDF) as f:
        return f.read()


def test_lidar_in_imu_is_upright_and_near(urdf):
    rot, trans = fixed_transform(urdf, 'livox_imu_link', 'livox_link')
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)))
    assert angle < 2.0
    assert np.linalg.norm(trans) < 0.1


def test_body_to_pelvis_puts_the_pelvis_below_the_lidar(urdf):
    rot, trans = fixed_transform(urdf, 'livox_imu_link', 'pelvis')
    # pelvis about 0.7 m down the torso; the IMU pitched forward under 20 deg
    assert -0.8 < trans[2] < -0.6
    assert np.degrees(np.arccos(rot[2, 2])) < 20.0
