"""
Test perception_utils' rotation helpers.

mat_to_quat comes from model_server.rotations, which graspgen_server shares. No ROS
graph is needed (geometry_msgs only).
"""

import math

import numpy as np
import pytest

from golem_skills.perception_utils import (mat_to_quat, matrix_to_pose, pose_to_matrix,
                                           quat_geodesic, quat_to_rot)


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
