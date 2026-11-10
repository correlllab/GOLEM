"""
Test the H1-2 URDF's MID-360 frames, which config/h12/fastlio.yaml's extrinsic is taken from.

livox_link and livox_imu_link are the MID-360's raw frames: the sensor hangs upside
down under the head, and the driver applies no rotation, so both frames point their
z axis at the floor. FAST-LIO's extrinsic (the lidar in the IMU frame) stays near
identity because the points and the IMU share that inversion.
"""

import glob
import os
import xml.etree.ElementTree as ET

from golem_slam.robot_geometry import rpy_to_matrix
import numpy as np
import pytest
from urdf_chain import link_transform

URDF = os.environ.get('H12_URDF', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'CL_Assets', 'ros_assets',
    'h1_2_magpie_ros.urdf'))

# The point frame and the IMU chip as an upright sensor would have them (m, rad, URDF rpy):
# livox_joint on torso_link and livox_imu_joint on livox_link. The raw frames are these
# turned 180 deg about their own x axis.
UPRIGHT_LIVOX = ((0.04874, 0.0, 0.67980), (0.0, 0.1531, 0.0))
UPRIGHT_IMU = ((0.011, -0.02329, 0.04412), (-0.0016, 0.0119, 0.0))
RX_PI = rpy_to_matrix(np.pi, 0.0, 0.0)
DOWN_WITHIN_20_DEG = -np.cos(np.radians(20.0))


@pytest.fixture(scope='module')
def urdf():
    """Return the H1-2 ROS URDF's text, skipping when the checkout has none."""
    if not os.path.isfile(URDF):
        pytest.skip(f'no URDF at {URDF}')
    with open(URDF) as f:
        return f.read()


def _variants():
    """Return every H1-2 URDF beside URDF that carries the MID-360."""
    paths = []
    for path in sorted(glob.glob(os.path.join(os.path.dirname(URDF), 'h1_2_*.urdf'))):
        with open(path) as f:
            if 'name="livox_link"' in f.read():
                paths.append(path)
    return paths


def _root(urdf_xml):
    """Return the URDF's root link."""
    root = ET.fromstring(urdf_xml)
    children = {j.find('child').get('link') for j in root.findall('joint')}
    return next(link.get('name') for link in root.findall('link')
                if link.get('name') not in children)


def _pose(xyz, rpy):
    """Return the 4x4 transform of a URDF origin."""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def test_lidar_in_imu_is_near_identity(urdf):
    """The lidar in the IMU frame is within 2 deg and 0.1 m of identity."""
    t = link_transform(urdf, 'livox_imu_link', 'livox_link')
    rot, trans = t[:3, :3], t[:3, 3]
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)))
    assert angle < 2.0
    assert np.linalg.norm(trans) < 0.1


def test_imu_is_upside_down_in_the_pelvis(urdf):
    """The IMU's z axis points at the floor, tilted only by the mount's pitch (about 9.5 deg)."""
    rot = link_transform(urdf, 'pelvis', 'livox_imu_link')[:3, :3]
    assert rot[2, 2] < DOWN_WITHIN_20_DEG


def test_body_to_pelvis_puts_the_pelvis_on_the_imu_minus_z_side(urdf):
    """The pelvis is about 0.7 m down the torso, which is +z in the upside-down IMU frame."""
    t = link_transform(urdf, 'livox_imu_link', 'pelvis')
    assert 0.6 < t[2, 3] < 0.8
    assert t[2, 2] < DOWN_WITHIN_20_DEG


def test_every_variant_turns_the_upright_frames_180_deg_about_x():
    """Every H1-2 URDF with the MID-360 has its frames turned 180 deg about x from upright."""
    variants = _variants()
    if not variants:
        pytest.skip(f'no H1-2 URDF with livox_link beside {URDF}')
    for path in variants:
        with open(path) as f:
            # The sphere models carry Drake tags without declaring their namespace.
            urdf_xml = f.read().replace('drake:', 'drake_')
        root = _root(urdf_xml)
        torso = link_transform(urdf_xml, root, 'torso_link')
        livox = torso @ _pose(*UPRIGHT_LIVOX)
        imu = livox @ _pose(*UPRIGHT_IMU)
        for link, upright in (('livox_link', livox), ('livox_imu_link', imu)):
            got = link_transform(urdf_xml, root, link)
            name = f'{os.path.basename(path)} {link}'
            np.testing.assert_allclose(got[:3, :3], upright[:3, :3] @ RX_PI, atol=1e-9,
                                       err_msg=name)
            np.testing.assert_allclose(got[:3, 3], upright[:3, 3], atol=1e-9, err_msg=name)
