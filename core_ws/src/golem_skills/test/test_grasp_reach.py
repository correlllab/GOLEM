"""
Test the grasp reach check: the hand-in-scene count, the approach waypoints and the IK replay.

The IK tests build golem_upper_ik's H1-2 model once (~17 s) and need no ROS graph.
"""

import math

import numpy as np
import pytest

from golem_skills.grasp_synthesis.common import level_pose, pose_at
from golem_skills.perception_utils import approach_pose, pose_to_matrix, pose_xyz
from golem_skills.skills.grasp import (APPROACH_LIN_TOL, AXIS_APPROACH_STEP_M,
                                       LINEAR_APPROACH_SEG_M, SERVO_LIN_TOL,
                                       approach_waypoints, line_waypoints)
from golem_skills.skills.grasp_reach import (HAND_HIT_POINTS, OBJECT_CLEAR_M, SCENE_CROP_M,
                                             ReachModel, hand_hits, scene_obstacles)


def plane(z, half=0.3, step=0.005):
    """Return an (N, 3) grid of pelvis-frame points at height `z` around the origin."""
    xs, ys = np.meshgrid(np.arange(-half, half, step), np.arange(-half, half, step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=-1)


def test_a_hand_above_a_surface_is_clear_and_one_through_it_is_not():
    """A level hand 5 cm above a table holds no table points; one at table height holds many."""
    table = plane(0.0)
    level = pose_to_matrix(level_pose((0.0, 0.0, 0.05), (1.0, 0.0, 0.0)))
    assert hand_hits(table, level) == 0
    level[2, 3] = 0.0
    assert hand_hits(table, level) >= HAND_HIT_POINTS


def test_a_top_down_hand_whose_pads_straddle_a_part_is_clear():
    """The pads beyond the contact plane are untested: a top-down grasp on a flat part passes."""
    table = plane(0.0)
    down = np.eye(4)
    down[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # contact point 5 mm above the table: the base sits 0.1146 m above it
    down[:3, 3] = (0.0, 0.0, 0.005 + 0.1146)
    assert hand_hits(table, down) == 0


def test_scene_obstacles_drops_the_object_and_far_points():
    """Points on the object and beyond SCENE_CROP_M are not obstacles."""
    obj = plane(0.0, half=0.02, step=0.002)
    near = np.array([[0.0, 0.0, OBJECT_CLEAR_M / 2], [0.1, 0.0, 0.0],
                     [SCENE_CROP_M + 0.05, 0.0, 0.0]])
    out = scene_obstacles(near, obj)
    np.testing.assert_allclose(out, [[0.1, 0.0, 0.0]])
    assert scene_obstacles(None, obj) is None


def test_line_waypoints_end_on_the_target_within_the_step():
    """A 25 cm line in 10 cm steps has three waypoints, the last the target."""
    wps = line_waypoints(pose_at((0.0, 0.0, 0.0)), pose_at((0.25, 0.0, 0.0)),
                         LINEAR_APPROACH_SEG_M, math.radians(20.0))
    assert len(wps) == 3
    np.testing.assert_allclose(pose_xyz(wps[-1]), (0.25, 0.0, 0.0))


def test_approach_waypoints_judge_the_pregrasp_and_the_first_quarter_of_the_axis():
    """The pre-grasp is judged, and the axis steps only until a close may happen."""
    cur = level_pose((0.10, 0.30, 0.20), (1.0, 0.0, 0.0))
    grasp = level_pose((0.45, 0.20, 0.0), (1.0, 0.0, 0.0))
    pregrasp = approach_pose(grasp, -0.10)
    contact = approach_pose(grasp, -0.025)
    wps = approach_waypoints(cur, pregrasp, contact)
    labels = [w[1] for w in wps]
    assert labels[0] == 'take-off'
    pre = wps[labels.index('pre-grasp')]
    assert pre[2][0] == APPROACH_LIN_TOL and pre[3]
    axis = [w for w in wps if w[1].startswith('axis')]
    assert len(axis) == math.ceil(0.075 / AXIS_APPROACH_STEP_M)
    assert axis[0][2][0] == SERVO_LIN_TOL and axis[-1][2] is None
    assert all(w[3] for w in axis)
    assert not any(w[3] for w in wps if w[1].startswith(('take-off', 'transit')))


@pytest.fixture(scope='module')
def reach():
    """Return a ReachModel with the arms at 'prep'."""
    model = ReachModel()
    q = np.zeros(model.model.num_motor)
    q[model.model.reduced_mask] = model.profile.named_configs['prep']
    return model, q


def _follow(reach, frame, cur, pregrasp, contact):
    """Return the first judged waypoint the IK misses, or None."""
    model, q = reach
    model.start(q)
    for pose, label, tol, _ in approach_waypoints(cur, pregrasp, contact):
        lin, ang = model.follow(frame, pose_to_matrix(pose))
        if tol is not None and (lin > tol[0] or ang > tol[1]):
            return label
    return None


def _cur(reach, frame):
    """Return `frame`'s pose at 'prep' as a Pose."""
    from golem_skills.perception_utils import matrix_to_pose
    model, q = reach
    model.start(q)
    return matrix_to_pose(model.frame_pose(frame))


def test_a_forward_grasp_in_front_of_the_left_hand_is_reachable(reach):
    """A level grasp pitched 30 deg down in front of the left hand passes."""
    frame = 'left_graspgenx_frame'
    a = np.array([math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30))])
    grasp = level_pose((0.40, 0.25, 0.05), a)
    assert _follow(reach, frame, _cur(reach, frame), approach_pose(grasp, -0.10),
                   approach_pose(grasp, -0.025)) is None


def test_a_level_side_grasp_at_the_support_height_is_not(reach):
    """A level left-hand grasp toward -Y at pelvis height, as on the flat bus bar, fails."""
    frame = 'left_graspgenx_frame'
    grasp = level_pose((0.47, 0.28, -0.015), (0.0, -1.0, 0.0))
    assert _follow(reach, frame, _cur(reach, frame), approach_pose(grasp, -0.10),
                   approach_pose(grasp, -0.025)) is not None


def test_a_cross_body_grasp_is_not(reach):
    """The left hand cannot come at a part from the robot's right."""
    frame = 'left_graspgenx_frame'
    grasp = level_pose((0.45, 0.05, 0.0), (0.0, 1.0, 0.0))
    assert _follow(reach, frame, _cur(reach, frame), approach_pose(grasp, -0.10),
                   approach_pose(grasp, -0.025)) is not None
