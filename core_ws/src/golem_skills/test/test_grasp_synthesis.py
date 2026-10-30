"""Unit tests for golem_skills.grasp_synthesis: every method on synthetic clouds,
no ROS graph needed (geometry_msgs only)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from geometry_msgs.msg import PoseStamped

from golem_skills.grasp_synthesis import (DEFAULT_METHOD, METHODS,
                                          SynthesisInput, method_name)
from golem_skills.grasp_synthesis.common import (GRIPPER_BASE_TO_CONTACT_M,
                                                 approach_from_az_pitch,
                                                 level_pose)
from golem_skills.grasp_synthesis.ranking import CROSS_BODY_TIER
from golem_skills.perception_utils import pose_to_matrix


class _Log:
    def info(self, *_):
        pass

    warn = error = info


def _box_cloud(center, size, n=2000, seed=0):
    """Points filling an axis-aligned box (pelvis frame)."""
    rng = np.random.default_rng(seed)
    return (np.asarray(center) +
            (rng.random((n, 3)) - 0.5) * np.asarray(size)).astype(np.float32)


def _fake_planner(poses, scores=None, width=0.05):
    calls = []

    def plan_grasp(cloud, gripper_name, frame, scene_cloud=None, arm=''):
        calls.append(dict(frame=frame, arm=arm))
        grasps = []
        for p in poses:
            g = PoseStamped()
            g.header.frame_id = frame
            g.pose = p
            grasps.append(g)
        return SimpleNamespace(
            grasps=grasps,
            scores=scores if scores is not None else
            list(np.linspace(0.9, 0.1, len(poses))),
            gripper_width=width)
    plan_grasp.calls = calls
    return plan_grasp


def _input(cloud, arm='right', min_pitch=0.0, planner=None):
    return SynthesisInput(obj='thing', arm=arm, cloud=cloud, scene=None,
                          min_pitch_deg=min_pitch,
                          plan_grasp=planner or _fake_planner([]), log=_Log())


def _contact(pose):
    T = pose_to_matrix(pose)
    return T[:3, 3] + GRIPPER_BASE_TO_CONTACT_M * T[:3, 2]


def _assert_valid(cands):
    assert cands.grasps
    n = len(cands.grasps)
    assert len(cands.scores) == len(cands.labels) == len(cands.tiers) == n
    assert cands.total >= n
    for g in cands.grasps:
        assert g.header.frame_id == 'pelvis'
        R = pose_to_matrix(g.pose)[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


# A forward reachable set for the planner-based methods: level-wrist
# approaches at a few azimuths/pitches, plus one mis-rolled and one
# cross-body grasp.
OBJ = np.array([0.45, -0.20, 0.05])


def _planner_pool():
    poses = [level_pose(OBJ, approach_from_az_pitch(az, p))
             for az, p in ((0, 30), (30, 30), (60, 10), (0, 10))]
    rolled = level_pose(OBJ, approach_from_az_pitch(10, 15))
    T = pose_to_matrix(rolled)
    T[:3, :2] = T[:3, :2] @ np.array([[0.0, -1.0], [1.0, 0.0]])  # roll 90 deg
    from golem_skills.perception_utils import matrix_to_pose
    poses.insert(1, matrix_to_pose(T))
    poses.append(level_pose(OBJ, approach_from_az_pitch(-120, 20)))
    return poses


def test_registry():
    assert DEFAULT_METHOD in METHODS
    assert set(METHODS) == {'heuristic', 'top_down', 'centroid', 'antipodal',
                            'graspgenx', 'graspgenx_filtered'}
    assert [m for m in METHODS.values() if m.staged] == [METHODS['top_down']]


def test_goal_top_down_overrides_the_parameter():
    assert method_name(True, 'antipodal') == 'top_down'
    assert method_name(False, ' antipodal ') == 'antipodal'


@pytest.mark.parametrize('name', sorted(METHODS))
def test_every_method_produces_valid_candidates(name):
    cloud = _box_cloud(OBJ, (0.04, 0.10, 0.06))
    cands, err = METHODS[name].synthesize(
        _input(cloud, planner=_fake_planner(_planner_pool())))
    assert err is None
    _assert_valid(cands)


def test_top_down_contact_on_centroid():
    cloud = _box_cloud(OBJ, (0.03, 0.03, 0.03))
    cands, _ = METHODS['top_down'].synthesize(_input(cloud))
    assert len(cands.grasps) == 1
    pose = cands.grasps[0].pose
    np.testing.assert_allclose(_contact(pose), cloud.mean(axis=0), atol=1e-6)
    R = pose_to_matrix(pose)[:3, :3]
    assert math.degrees(math.asin(-R[2, 2])) == pytest.approx(80.0, abs=1e-6)
    np.testing.assert_allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-6)


def test_heuristic_level_wrist_and_tier_order():
    cloud = _box_cloud(OBJ, (0.05, 0.05, 0.05))
    cands, _ = METHODS['heuristic'].synthesize(_input(cloud))
    assert cands.tiers == sorted(cands.tiers)
    assert CROSS_BODY_TIER not in cands.tiers
    for g in cands.grasps:
        R = pose_to_matrix(g.pose)[:3, :3]
        assert abs(R[2, 0]) < 1e-6      # closing axis horizontal
        assert R[2, 1] >= 0.0           # +Y up
        np.testing.assert_allclose(_contact(g.pose), cloud.mean(axis=0),
                                   atol=1e-6)


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_heuristic_mirrors_per_arm(arm):
    cloud = _box_cloud(OBJ, (0.05, 0.05, 0.05))
    cands, _ = METHODS['heuristic'].synthesize(_input(cloud, arm=arm))
    sign = 1.0 if arm == 'right' else -1.0
    for g in cands.grasps:
        ay = pose_to_matrix(g.pose)[1, 2]
        assert sign * ay >= -1e-6       # never approaches from across the body


def test_centroid_closes_across_pelvis_y():
    cloud = _box_cloud(OBJ, (0.10, 0.02, 0.04))
    cands, _ = METHODS['centroid'].synthesize(_input(cloud))
    first = pose_to_matrix(cands.grasps[0].pose)[:3, :3]
    assert abs(first[1, 0]) > 0.9       # closing axis ~ pelvis Y
    assert cands.labels[-1] == 'centroid-x-45deg'


@pytest.mark.parametrize('long_axis', ['x', 'y'])
def test_antipodal_closes_across_minor_axis(long_axis):
    size = (0.12, 0.03, 0.04) if long_axis == 'x' else (0.03, 0.12, 0.04)
    cloud = _box_cloud(OBJ, size)
    cands, _ = METHODS['antipodal'].synthesize(_input(cloud))
    closing = pose_to_matrix(cands.grasps[0].pose)[:3, 0]
    minor = np.array([0.0, 1.0, 0.0]) if long_axis == 'x' else np.array([1.0, 0.0, 0.0])
    horiz = closing[:2] / np.linalg.norm(closing[:2])
    assert abs(float(horiz @ minor[:2])) > 0.95
    assert cands.gripper_width == pytest.approx(0.03, abs=0.005)
    assert cands.labels[0].startswith('antipodal-minor')


def test_graspgenx_raw_executes_server_order_untouched():
    pool = _planner_pool()
    planner = _fake_planner(pool)
    cands, _ = METHODS['graspgenx'].synthesize(_input(
        _box_cloud(OBJ, (0.04, 0.04, 0.04)), planner=planner))
    assert len(cands.grasps) == 5           # MAX_GRASP_ATTEMPTS
    for g, p in zip(cands.grasps, pool):
        np.testing.assert_allclose(pose_to_matrix(g.pose), pose_to_matrix(p),
                                   atol=1e-9)
    assert cands.scores == sorted(cands.scores, reverse=True)
    assert planner.calls[0] == dict(frame='pelvis', arm='right')


def test_graspgenx_filtered_rerolls_and_drops_cross_body():
    pool = _planner_pool()
    cands, _ = METHODS['graspgenx_filtered'].synthesize(_input(
        _box_cloud(OBJ, (0.04, 0.04, 0.04)), planner=_fake_planner(pool)))
    assert CROSS_BODY_TIER not in cands.tiers
    assert cands.tiers == sorted(cands.tiers)
    assert 'ggx-5' not in cands.labels       # the cross-body grasp
    assert 'ggx-1' in cands.labels           # the mis-rolled grasp, re-rolled
    for g in cands.grasps:
        R = pose_to_matrix(g.pose)[:3, :3]
        assert abs(R[2, 0]) < 1e-6
        assert R[2, 1] >= 0.0


@pytest.mark.parametrize('name', sorted(METHODS))
def test_min_pitch_gate(name):
    cloud = _box_cloud(OBJ, (0.04, 0.04, 0.04))
    cands, err = METHODS[name].synthesize(
        _input(cloud, min_pitch=25.0, planner=_fake_planner(_planner_pool())))
    if err is not None:
        assert 'pitched' in err
        return
    for g in cands.grasps:
        assert -pose_to_matrix(g.pose)[2, 2] >= math.sin(math.radians(25.0)) - 1e-9


def test_planner_failure_is_a_reason_not_an_exception():
    def no_plan(*_, **__):
        return None
    for name in ('graspgenx', 'graspgenx_filtered'):
        cands, err = METHODS[name].synthesize(
            _input(_box_cloud(OBJ, (0.04, 0.04, 0.04)), planner=no_plan))
        assert cands is None and 'no grasp planned' in err
