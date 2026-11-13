"""
Test perception_utils' rotation, pose and Gemini-reply helpers.

mat_to_quat comes from model_server.rotations, which graspgen_server shares. No ROS
graph is needed (geometry_msgs only).
"""

import math

import numpy as np
import pytest

from golem_skills.perception_utils import (approach_pose, gemini_boxes, mat_to_quat,
                                           matrix_to_pose, pose_quat, pose_to_matrix,
                                           pose_xyz, quat_geodesic, quat_slerp, quat_to_rot)


def _rotation(axis, angle):
    """Return the 3x3 rotation of angle rad about the unit vector axis (Rodrigues)."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


# one rotation per branch of Shepperd's method: positive trace, then the largest
# diagonal entry on x, y and z (half turns)
BRANCHES = [_rotation((1, 2, 3), 0.7), _rotation((1, 0.1, 0.1), math.pi),
            _rotation((0.1, 1, 0.1), math.pi), _rotation((0.1, 0.1, 1), math.pi)]


@pytest.mark.parametrize('rotation', BRANCHES, ids=['trace', 'x', 'y', 'z'])
def test_mat_to_quat_inverts_quat_to_rot(rotation):
    """mat_to_quat gives a unit (x, y, z, w) quaternion that quat_to_rot maps back."""
    quat = mat_to_quat(rotation)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    np.testing.assert_allclose(quat_to_rot(*quat), rotation, atol=1e-12)


def test_random_rotations_round_trip_through_a_pose():
    """matrix_to_pose and pose_to_matrix are inverses, and the quaternion keeps its angle."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        transform = np.eye(4)
        transform[:3, :3] = _rotation(rng.normal(size=3), rng.uniform(0.0, math.pi))
        transform[:3, 3] = rng.normal(size=3)
        np.testing.assert_allclose(pose_to_matrix(matrix_to_pose(transform)), transform,
                                   atol=1e-12)


def test_quat_geodesic_is_the_rotation_angle_and_sign_invariant():
    """quat_geodesic is the angle between two rotations, whatever either quaternion's sign."""
    a = np.array(mat_to_quat(np.eye(3)))
    b = np.array(mat_to_quat(_rotation((0, 0, 1), 0.5)))
    assert quat_geodesic(a, b) == pytest.approx(0.5)
    assert quat_geodesic(a, -b) == pytest.approx(0.5)
    assert quat_geodesic(b, b) == pytest.approx(0.0, abs=1e-7)


def test_quat_slerp_ends_at_both_quaternions_and_halves_the_angle():
    """quat_slerp starts at q0, ends at q1 and is halfway along the arc at t = 0.5."""
    a = np.array(mat_to_quat(np.eye(3)))
    b = np.array(mat_to_quat(_rotation((0, 1, 0), 1.2)))
    assert quat_geodesic(quat_slerp(a, b, 0.0), a) == pytest.approx(0.0, abs=1e-7)
    assert quat_geodesic(quat_slerp(a, b, 1.0), b) == pytest.approx(0.0, abs=1e-7)
    assert quat_geodesic(quat_slerp(a, b, 0.5), a) == pytest.approx(0.6)


def test_quat_slerp_takes_the_short_way_round():
    """A quaternion and its negation are the same rotation, so slerp never takes the long arc."""
    a = np.array(mat_to_quat(np.eye(3)))
    b = np.array(mat_to_quat(_rotation((0, 0, 1), 0.4)))
    assert quat_geodesic(quat_slerp(a, -b, 0.5), a) == pytest.approx(0.2)


def test_approach_pose_moves_along_the_pose_z_axis_and_keeps_orientation():
    """approach_pose shifts the position along the pose's own +Z and leaves the input alone."""
    transform = np.eye(4)
    transform[:3, :3] = _rotation((1, 0, 0), math.pi / 2)   # +Z points along -Y
    transform[:3, 3] = (0.3, 0.1, 0.2)
    pose = matrix_to_pose(transform)
    deeper = approach_pose(pose, 0.05)
    np.testing.assert_allclose(pose_xyz(deeper), (0.3, 0.05, 0.2), atol=1e-12)
    np.testing.assert_allclose(pose_quat(deeper), pose_quat(pose))
    np.testing.assert_allclose(pose_xyz(approach_pose(pose, -0.1)), (0.3, 0.2, 0.2), atol=1e-12)
    np.testing.assert_allclose(pose_xyz(pose), (0.3, 0.1, 0.2))


def test_gemini_boxes_scales_and_sorts_every_box():
    """Normalized [y1, x1, y2, x2] boxes become pixel [x1, y1, x2, y2], sorted, in order."""
    reply = ('```json\n[{"box_2d": [100, 200, 300, 400], "label": "nut"},'
             ' {"box_2d": [500, 900, 250, 600]}]\n```')
    assert gemini_boxes(reply, 640, 360) == [
        pytest.approx([128.0, 36.0, 256.0, 108.0]),
        pytest.approx([384.0, 90.0, 576.0, 180.0])]


def test_gemini_boxes_takes_one_object_and_skips_bad_entries():
    """A single object is one box; entries without a 4-number box_2d are skipped."""
    assert gemini_boxes('{"box_2d": [0, 0, 1000, 1000]}', 100, 50) == [
        pytest.approx([0.0, 0.0, 100.0, 50.0])]
    reply = '[{"box_2d": [1, 2, 3]}, {"label": "x"}, "nut", {"box_2d": [0, 0, 500, 500]}]'
    assert gemini_boxes(reply, 100, 100) == [pytest.approx([0.0, 0.0, 50.0, 50.0])]


def test_gemini_boxes_is_empty_without_a_box():
    """An empty, non-JSON or box-free reply gives no boxes."""
    for reply in ('', None, 'no nut here', '[]', '{}'):
        assert gemini_boxes(reply, 640, 360) == []
