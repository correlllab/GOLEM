"""
Test the grasp reach check: the hand-in-scene count, the approach waypoints and the IK replay.

The IK tests build golem_upper_ik's H1-2 model (~15 s) once in this process and once in the
check's child process, and its G1 model once, and need no ROS graph.
"""

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from golem_skills.grasp_synthesis.common import grasp_pose, level_pose, pose_at
from golem_skills.perception_utils import approach_pose, pose_to_matrix, pose_xyz
from golem_skills.skills.grasp import (APPROACH_LIN_TOL, AXIS_APPROACH_STEP_M,
                                       LINEAR_APPROACH_SEG_M, SERVO_LIN_TOL,
                                       approach_waypoints, line_waypoints)
from golem_skills.reach_model import (HAND_HIT_POINTS, SUPPORT_DEPTH_M, ReachModel,
                                      hand_hits, support_depth, support_edges)
from golem_skills.robots import robot_profile
from golem_skills.skills.grasp_reach import (OBJECT_CLEAR_M, SCENE_CROP_M, GraspReach,
                                             scene_obstacles)


def plane(z, half=0.3, step=0.005):
    """Return an (N, 3) grid of pelvis-frame points at height `z` around the origin."""
    xs, ys = np.meshgrid(np.arange(-half, half, step), np.arange(-half, half, step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=-1)


def test_a_hand_above_a_surface_is_clear_and_one_through_it_is_not():
    """A level hand 5 cm above a table holds no table points; one at table height holds many."""
    table = plane(0.0)
    level = pose_to_matrix(level_pose((0.0, 0.0, 0.05), (1.0, 0.0, 0.0)))
    assert hand_hits(table, level, 'magpie') == 0
    level[2, 3] = 0.0
    assert hand_hits(table, level, 'magpie') >= HAND_HIT_POINTS


def test_a_top_down_hand_whose_pads_straddle_a_part_is_clear():
    """The pads beyond the contact plane are untested: a top-down grasp on a flat part passes."""
    table = plane(0.0)
    down = np.eye(4)
    down[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # contact point 5 mm above the table: the base sits 0.1146 m above it
    down[:3, 3] = (0.0, 0.0, 0.005 + 0.1146)
    assert hand_hits(table, down, 'magpie') == 0


def seen_support(x0=0.38, y0=-0.3, y1=0.3):
    """Return a table top at pelvis z 0 that the camera sees from `x0` on, between y0 and y1."""
    xs, ys = np.meshgrid(np.arange(x0, 0.6, 0.005), np.arange(y0, y1, 0.005))
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=-1)


def rolled_pregrasp(z):
    """Return the base pose 10 cm back of a +X grasp at (0.45, 0, z) closing vertically."""
    grasp = grasp_pose((0.45, 0.0, z), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    return pose_to_matrix(approach_pose(grasp, -0.10))


def test_a_rolled_hand_below_the_unseen_table_front_is_caught():
    """A vertical-closing hand in front of the seen table, fingers below its top, fails."""
    table = seen_support()
    T = rolled_pregrasp(0.02)
    assert hand_hits(table, T, 'magpie') == 0
    assert support_depth(support_edges(table), T, 'magpie') > 0.04


def test_a_level_hand_above_the_unseen_table_front_is_clear():
    """A level hand in front of the seen table, above its top, passes."""
    table = seen_support()
    level = pose_to_matrix(approach_pose(level_pose((0.45, 0.0, 0.05), (1.0, 0.0, 0.0)), -0.10))
    assert support_depth(support_edges(table), level, 'magpie') <= SUPPORT_DEPTH_M


def test_the_support_extends_only_in_the_slices_it_is_seen_in():
    """The rolled hand passes when the table is seen only beside it, not ahead of it."""
    table = seen_support(y0=0.15)
    depth = support_depth(support_edges(table), rolled_pregrasp(0.02), 'magpie')
    assert depth <= SUPPORT_DEPTH_M


def test_a_hand_behind_the_seen_near_edge_is_left_to_the_point_count():
    """Hand points at or past a slice's near edge are not tested against the extension."""
    table = seen_support(x0=0.20)
    assert support_depth(support_edges(table), rolled_pregrasp(0.02), 'magpie') == 0.0


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
    model = ReachModel('h12')
    q = np.zeros(model.model.num_motor)
    q[model.model.reduced_mask] = model.profile.named_configs['prep']
    return model, q


def _follow(reach, frame, cur, pregrasp, contact):
    """Return why the IK misses the approach without a scene, or None."""
    model, q = reach
    waypoints = [(pose_to_matrix(p), label, tol, scene)
                 for p, label, tol, scene in approach_waypoints(cur, pregrasp, contact)]
    return model.check(q, frame, waypoints, None)


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


def test_the_check_rejects_a_waypoint_below_the_extended_support(reach):
    """check() fails a scene waypoint whose hand sits below the support ahead of it."""
    model, q = reach
    T = rolled_pregrasp(0.02)
    T[1, 3] = 0.25
    table = seen_support(y0=0.1, y1=0.4)
    why = model.check(q, 'left_graspgenx_frame', [(T, 'pre-grasp', None, True)], table)
    assert why is not None and 'below the support' in why
    assert model.check(q, 'left_graspgenx_frame', [(T, 'pre-grasp', None, False)], table) is None


def test_a_cross_body_grasp_is_not(reach):
    """The left hand cannot come at a part from the robot's right."""
    frame = 'left_graspgenx_frame'
    grasp = level_pose((0.45, 0.05, 0.0), (0.0, 1.0, 0.0))
    assert _follow(reach, frame, _cur(reach, frame), approach_pose(grasp, -0.10),
                   approach_pose(grasp, -0.025)) is not None


class _Node(GraspReach):
    """Stand in for the skills node: a logger and a fixed /joint_states."""

    def __init__(self, joint_state, robot='h12'):
        """Keep `joint_state` and collect log lines."""
        self.joint_state = joint_state
        self.robot = robot_profile(robot)
        self.lines = []

    def get_logger(self):
        """Return self, which logs into `lines`."""
        return self

    def info(self, msg):
        """Collect an info line."""
        self.lines.append(msg)

    warn = error = info

    def latest_joint_state(self):
        """Return the fixed joint state."""
        return self.joint_state


def test_the_child_process_check_leaves_the_node_threads_running(reach):
    """While the model builds and checks, another thread runs every 0.5 s; the verdicts hold."""
    model, q = reach
    node = _Node(SimpleNamespace(name=model.joints, position=list(q)))
    gaps, stop = [], threading.Event()

    def tick():
        """Record the gap between wake-ups every 0.01 s until `stop` is set."""
        last = time.monotonic()
        while not stop.wait(0.01):
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        frame = 'left_graspgenx_frame'
        cur = _cur(reach, frame)
        side = level_pose((0.47, 0.28, -0.015), (0.0, -1.0, 0.0))
        a = np.array([math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30))])
        forward = level_pose((0.40, 0.25, 0.05), a)
        for grasp, fails in ((forward, False), (side, True)):
            wps = approach_waypoints(cur, approach_pose(grasp, -0.10),
                                     approach_pose(grasp, -0.025))
            assert (node.reach_check(frame, wps, None) is not None) == fails, node.lines
    finally:
        stop.set()
        ticker.join()
        if node._reach_proc is not None:
            node._reach_proc.kill()
    assert any('IK model built' in line for line in node.lines), node.lines
    assert max(gaps) < 0.5


def test_a_dex3_hand_above_a_surface_is_clear_and_one_through_it_is_not():
    """A level Dex3-1 with its contact 6 cm above a table holds no table points; at it, many."""
    table = plane(0.0)
    level = pose_to_matrix(level_pose((0.0, 0.0, 0.06), (1.0, 0.0, 0.0)))
    assert hand_hits(table, level, 'dex3') == 0
    level[2, 3] -= 0.06
    assert hand_hits(table, level, 'dex3') >= HAND_HIT_POINTS


@pytest.fixture(scope='module')
def reach_g1():
    """Return the G1's ReachModel with the arms at 'home'."""
    model = ReachModel('g1')
    q = np.zeros(model.model.num_motor)
    q[model.model.reduced_mask] = model.profile.named_configs['home']
    return model, q


def test_the_g1_ik_follows_its_tool_frame_through_the_palm(reach_g1):
    """From 'home' the left Dex3-1 grasp frame reaches 5 cm ahead, not 1 m ahead."""
    model, q = reach_g1
    frame = 'left_dex3_grasp_frame'
    model.start(q)
    cur = model.frame_pose('left_hand_palm_link') @ model.robot.tool_frames[frame].matrix()
    near, far = cur.copy(), cur.copy()
    near[0, 3] += 0.05
    far[0, 3] += 1.0
    tol = (APPROACH_LIN_TOL, 0.10)
    assert model.check(q, frame, [(near, 'pre-grasp', tol, False)], None) is None
    why = model.check(q, frame, [(far, 'pre-grasp', tol, False)], None)
    assert why is not None and 'short of the pre-grasp' in why
