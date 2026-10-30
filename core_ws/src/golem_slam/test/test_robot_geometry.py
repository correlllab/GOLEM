import numpy as np

from golem_slam.robot_geometry import (
    fixed_transform, link_boxes, matrix_to_quat, points_in_box, quat_to_matrix,
    rotation_aligning, rpy_to_matrix)

URDF = """<robot name="t">
  <link name="a"><collision><origin xyz="1 0 0" rpy="0 0 1.5707963"/>
    <geometry><box size="0.2 0.4 0.6"/></geometry></collision></link>
  <link name="b"><collision><geometry><cylinder radius="0.1" length="1"/></geometry>
  </collision></link>
  <link name="c"/>
</robot>"""


def test_link_boxes_apply_origin_and_padding():
    boxes = link_boxes(URDF, 0.01)
    assert set(boxes) == {'a', 'b'}
    lo, hi = boxes['a']
    # 0.2 x 0.4 box yawed 90 deg: extents swap in x/y.
    np.testing.assert_allclose(lo, [0.79, -0.11, -0.31], atol=1e-6)
    np.testing.assert_allclose(hi, [1.21, 0.11, 0.31], atol=1e-6)
    np.testing.assert_allclose(boxes['b'][1], [0.11, 0.11, 0.51], atol=1e-6)


def test_points_in_posed_box():
    lo, hi = np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1])
    rot = rpy_to_matrix(0.0, 0.0, np.pi / 4)
    trans = np.array([2.0, 0.0, 0.0])
    pts = np.array([[2.0, 0.0, 0.0], [2.12, 0.0, 0.0], [2.0, 0.13, 0.0], [0.0, 0.0, 0.0]])
    # A 45 deg yawed 0.2 m cube reaches 0.141 m along the frame axes.
    assert points_in_box(pts, rot, trans, lo, hi).tolist() == [True, True, True, False]


def test_rotation_aligning():
    for a in ([0.0, 0.0, 1.0], [0.24, 0.0, 0.97], [0.0, 0.0, -1.0], [1.0, 2.0, 3.0]):
        a = np.array(a) / np.linalg.norm(a)
        r = rotation_aligning(a, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(r @ a, [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(r), 1.0)


def test_quaternion_round_trip():
    for rpy in ((0.1, -0.24, 2.0), (3.1, 0.0, -3.1), (0.0, 1.5, 0.0)):
        r = rpy_to_matrix(*rpy)
        np.testing.assert_allclose(quat_to_matrix(*matrix_to_quat(r)), r, atol=1e-9)


CHAIN = """<robot name="t">
  <link name="base"/><link name="torso"/><link name="lidar"/><link name="imu"/>
  <joint name="waist" type="revolute"><origin xyz="0 0 0.1"/>
    <parent link="base"/><child link="torso"/><axis xyz="0 0 1"/></joint>
  <joint name="mount" type="fixed"><origin xyz="0.05 0 0.7" rpy="0 0.15 0"/>
    <parent link="torso"/><child link="lidar"/></joint>
  <joint name="chip" type="fixed"><origin xyz="0.01 -0.02 0.04" rpy="-0.002 0.012 0"/>
    <parent link="lidar"/><child link="imu"/></joint>
</robot>"""


def test_fixed_transform_composes_and_inverts():
    rot, trans = fixed_transform(CHAIN, 'base', 'imu')
    r_mount, r_chip = rpy_to_matrix(0.0, 0.15, 0.0), rpy_to_matrix(-0.002, 0.012, 0.0)
    np.testing.assert_allclose(rot, r_mount @ r_chip, atol=1e-12)
    expected = np.array([0.0, 0.0, 0.1]) + [0.05, 0.0, 0.7] + r_mount @ [0.01, -0.02, 0.04]
    np.testing.assert_allclose(trans, expected, atol=1e-12)
    # the reverse direction is the inverse
    rot_i, trans_i = fixed_transform(CHAIN, 'imu', 'lidar')
    np.testing.assert_allclose(rot_i, r_chip.T, atol=1e-12)
    np.testing.assert_allclose(trans_i, -r_chip.T @ [0.01, -0.02, 0.04], atol=1e-12)
