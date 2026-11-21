"""Test the per-robot profiles the manipulation skills read (robots.py)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from golem_skills.robots import (ARMS, CONVENTION_CONTACT_M, DEFAULT_ROBOT, ROBOTS,
                                 drive_pose, robot_profile)


def test_default_robot_is_the_h12():
    """A node started without `robot` drives the H1-2, as before the parameter existed."""
    assert DEFAULT_ROBOT == 'h12'
    p = robot_profile(DEFAULT_ROBOT)
    assert p.grasp_frames == {'left': 'left_graspgenx_frame',
                              'right': 'right_graspgenx_frame'}
    assert p.pinch_frames == {'left': 'left_grasp_frame', 'right': 'right_grasp_frame'}
    assert p.prep_config == 'prep'
    assert p.staging_configs == {'left': 'staging_left', 'right': 'staging_right'}
    assert p.graspgen_gripper == 'magpie'


def test_robot_names_are_case_and_space_insensitive():
    """' H12 ' selects the h12 profile."""
    assert robot_profile(' H12 ') is ROBOTS['h12']


def test_unknown_robot_is_refused():
    """An unknown name raises ValueError naming the robots there are."""
    with pytest.raises(ValueError, match='unknown robot'):
        robot_profile('h1')


@pytest.mark.parametrize('name', sorted(ROBOTS))
def test_every_profile_covers_both_arms(name):
    """Every per-arm table has exactly the arms a goal may name."""
    p = ROBOTS[name]
    for table in (p.grasp_frames, p.pinch_frames, p.staging_configs):
        assert tuple(sorted(table)) == tuple(sorted(ARMS))


def _refusal(profile, method, visual_servo=False):
    """Return GraspSkill._robot_refusal for `profile` on `method` and a goal's visual_servo."""
    from golem_skills.grasp_synthesis import METHODS
    from golem_skills.skills.grasp import GraspSkill
    node = SimpleNamespace(robot=profile)
    goal = SimpleNamespace(visual_servo=visual_servo)
    return GraspSkill._robot_refusal(node, METHODS[method], goal)


def test_the_h12_refuses_nothing():
    """The H1-2 runs GraspGenX and the visual servo."""
    h12 = robot_profile('h12')
    assert _refusal(h12, 'graspgenx_filtered', visual_servo=True) is None


def test_a_robot_without_graspgen_or_hand_cameras_is_refused_before_moving():
    """No GraspGenX gripper model refuses its methods; no hand cameras refuse visual_servo."""
    bare = replace(robot_profile('h12'), name='bare', graspgen_gripper=None,
                   hand_cameras=False)
    assert 'GraspGenX' in _refusal(bare, 'graspgenx')
    assert 'antipodal' in _refusal(bare, 'graspgenx_filtered')
    assert _refusal(bare, 'antipodal') is None
    assert 'hand cameras' in _refusal(bare, 'top_down', visual_servo=True)


def test_the_convention_contact_distance_is_grasp_synthesis_own():
    """robots.py's copy of the grasp convention's contact distance equals common.py's."""
    from golem_skills.grasp_synthesis.common import GRIPPER_BASE_TO_CONTACT_M
    assert CONVENTION_CONTACT_M == GRIPPER_BASE_TO_CONTACT_M


@pytest.mark.parametrize('arm', ARMS)
def test_g1_grasp_and_pinch_frames_follow_the_grasp_convention(arm):
    """Each Dex3-1 grasp frame approaches along the palm face's normal, its contact the pinch."""
    g1 = robot_profile('g1')
    grasp = g1.tool_frames[g1.grasp_frames[arm]]
    pinch = g1.tool_frames[g1.pinch_frames[arm]]
    assert grasp.parent == pinch.parent == f'{arm}_hand_palm_link'
    Tg, Tp = grasp.matrix(), pinch.matrix()
    R = Tg[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    side = -1.0 if arm == 'left' else 1.0
    # approach (+Z) along the palm face's normal, closing (+X) along the palm's +X
    np.testing.assert_allclose(R[:, 2], [0.0, side, 0.0], atol=1e-12)
    np.testing.assert_allclose(R[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    # the contact point, CONVENTION_CONTACT_M along +Z, is the pinch frame's origin
    np.testing.assert_allclose(Tg[:3, 3] + CONVENTION_CONTACT_M * R[:, 2], Tp[:3, 3],
                               atol=1e-12)
    np.testing.assert_allclose(Tp[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(Tp[:3, 3], [0.08, side * 0.04, 0.0], atol=1e-12)


def test_g1_hands_mirror():
    """The right hand's frames are the left's mirrored across the palm's XZ plane."""
    g1 = robot_profile('g1')
    M = np.diag([1.0, -1.0, 1.0])
    for kind in ('grasp', 'pinch'):
        left = g1.tool_frames[f'left_dex3_{kind}_frame'].matrix()
        right = g1.tool_frames[f'right_dex3_{kind}_frame'].matrix()
        np.testing.assert_allclose(M @ left[:3, 3], right[:3, 3], atol=1e-12)
        # the approach mirrors, the closing axis stays
        np.testing.assert_allclose(M @ left[:3, 2], right[:3, 2], atol=1e-12)
        np.testing.assert_allclose(left[:3, 0], right[:3, 0], atol=1e-12)


def test_drive_pose_goes_through_the_tool_frames_parent():
    """A tool frame is driven through its parent at T inv(offset); a URDF frame as given."""
    g1 = robot_profile('g1')
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = (0.4, 0.2, 0.1)
    frame, Tp = drive_pose(g1, 'left_dex3_grasp_frame', T)
    assert frame == 'left_hand_palm_link'
    np.testing.assert_allclose(Tp @ g1.tool_frames['left_dex3_grasp_frame'].matrix(), T,
                               atol=1e-12)
    assert drive_pose(g1, 'left_hand_palm_link', T) == ('left_hand_palm_link', T)
    assert drive_pose(robot_profile('h12'), 'left_graspgenx_frame', T)[0] == \
        'left_graspgenx_frame'


def test_g1_profile():
    """The G1 grasps with the Dex3-1, from 'home', with neither GraspGenX nor a visual servo."""
    g1 = robot_profile('g1')
    assert (g1.hand, g1.prep_config, g1.graspgen_gripper, g1.hand_cameras) == \
        ('dex3', 'home', None, False)
    assert set(g1.staging_configs.values()) == {'home'}
    assert 'graspgenx' in _refusal(g1, 'graspgenx')
    assert _refusal(g1, 'heuristic') is None
