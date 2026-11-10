"""Test urdf_chain, the link-pose helper the frame tests rely on."""
from golem_slam.robot_geometry import rpy_to_matrix
import numpy as np
import pytest
from urdf_chain import link_transform

CHAIN = """<robot name="t">
  <link name="base"/><link name="torso"/><link name="lidar"/><link name="imu"/>
  <link name="other"/>
  <joint name="waist" type="revolute"><origin xyz="0 0 0.1"/>
    <parent link="base"/><child link="torso"/><axis xyz="0 0 1"/></joint>
  <joint name="mount" type="fixed"><origin xyz="0.05 0 0.7" rpy="0 0.15 0"/>
    <parent link="torso"/><child link="lidar"/></joint>
  <joint name="chip" type="fixed"><origin xyz="0.01 -0.02 0.04" rpy="-0.002 0.012 0"/>
    <parent link="lidar"/><child link="imu"/></joint>
</robot>"""
R_MOUNT = rpy_to_matrix(0.0, 0.15, 0.0)
R_CHIP = rpy_to_matrix(-0.002, 0.012, 0.0)


def test_composes_and_inverts():
    """Fixed joints compose down the chain, and the reverse direction is the inverse."""
    t = link_transform(CHAIN, 'base', 'imu')
    np.testing.assert_allclose(t[:3, :3], R_MOUNT @ R_CHIP, atol=1e-12)
    expected = np.array([0.0, 0.0, 0.1]) + [0.05, 0.0, 0.7] + R_MOUNT @ [0.01, -0.02, 0.04]
    np.testing.assert_allclose(t[:3, 3], expected, atol=1e-12)
    back = link_transform(CHAIN, 'imu', 'lidar')
    np.testing.assert_allclose(back[:3, :3], R_CHIP.T, atol=1e-12)
    np.testing.assert_allclose(back[:3, 3], -R_CHIP.T @ [0.01, -0.02, 0.04], atol=1e-12)


def test_a_revolute_joint_turns_about_its_axis():
    """A revolute joint's position turns everything below it about the joint's axis."""
    turned = link_transform(CHAIN, 'base', 'lidar', {'waist': 0.5})
    np.testing.assert_allclose(turned[:3, :3], rpy_to_matrix(0.0, 0.0, 0.5) @ R_MOUNT,
                               atol=1e-12)
    np.testing.assert_allclose(turned[:3, 3],
                               [0.05 * np.cos(0.5), 0.05 * np.sin(0.5), 0.8], atol=1e-12)


def test_links_in_two_trees_are_refused():
    """Links that do not share a root are refused."""
    with pytest.raises(ValueError, match='not in one URDF tree'):
        link_transform(CHAIN, 'base', 'other')
