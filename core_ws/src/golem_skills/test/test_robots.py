"""Test the per-robot profiles the manipulation skills read (robots.py)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from golem_skills.robots import (ARMS, CONVENTION_CONTACT_M, DEFAULT_ROBOT, ROBOTS,
                                 arm_of, robot_profile)


def test_default_robot_is_the_h12():
    """A node started without `robot` drives the H1-2, as before the parameter existed."""
    assert DEFAULT_ROBOT == 'h12'
    p = robot_profile(DEFAULT_ROBOT)
    assert p.grasp_frames == {'left': 'left_graspgenx_frame',
                              'right': 'right_graspgenx_frame'}
    assert p.pinch_frames == {'left': 'left_grasp_frame', 'right': 'right_grasp_frame'}
    assert p.graspgen_gripper == 'magpie'
    # the grasping arm leaves prep for its staging posture, the other arm stays
    assert p.staging_postures['left']['right'] == p.prep_posture['right']
    assert p.staging_postures['right']['left'] == p.prep_posture['left']
    assert p.staging_postures['left']['left'] != p.prep_posture['left']


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
    for table in (p.grasp_frames, p.pinch_frames, p.prep_posture, p.staging_postures,
                  *p.staging_postures.values()):
        assert tuple(sorted(table)) == tuple(sorted(ARMS))


@pytest.mark.parametrize('name', sorted(ROBOTS))
def test_every_posture_is_a_pelvis_position_and_a_unit_quaternion(name):
    """Every posture pose is (x, y, z) in reach of the pelvis and a unit (qx, qy, qz, qw)."""
    p = ROBOTS[name]
    for posture in (p.prep_posture, *p.staging_postures.values()):
        for pose in posture.values():
            assert len(pose) == 7
            assert np.linalg.norm(pose[:3]) < 1.0
            assert np.linalg.norm(pose[3:]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize('name', sorted(ROBOTS))
def test_the_pinch_frames_are_golem_upper_iks_grasp_frames(name):
    """Each arm's pinch frame is the grasp frame golem_upper_ik's /grasp_frame_target drives."""
    from golem_upper_ik.profile import load_profile
    upper = load_profile(name)
    p = ROBOTS[name]
    assert [p.pinch_frames[arm] for arm in ARMS] == [f.name for f in upper.grasp_frames]


def test_arm_of_names_the_arm_of_a_grasp_or_pinch_frame():
    """arm_of maps each grasp and pinch frame to its arm, and any other frame to None."""
    h12 = robot_profile('h12')
    assert arm_of(h12, 'left_graspgenx_frame') == 'left'
    assert arm_of(h12, 'right_grasp_frame') == 'right'
    assert arm_of(h12, 'pelvis') is None


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


def _old_dex3_frames(arm):
    """
    Return the Dex3-1 frames golem_skills defined on the palm before golem_upper_ik had them.

    (grasp frame, pinch frame), each a 4x4 pose in <arm>_hand_palm_link: the pinch point
    0.08 m along the palm's +X and 0.04 m in front of its face, the approach (+Z) the face's
    normal, the closing axis (+X) the palm's +X, the grasp frame CONVENTION_CONTACT_M behind
    the pinch point.
    """
    side = -1.0 if arm == 'left' else 1.0
    z = np.array([0.0, side, 0.0])
    x = np.array([1.0, 0.0, 0.0])
    R = np.column_stack([x, np.cross(z, x), z])
    contact = np.array([0.08, side * 0.04, 0.0])
    grasp, pinch = np.eye(4), np.eye(4)
    grasp[:3, :3] = pinch[:3, :3] = R
    grasp[:3, 3] = contact - CONVENTION_CONTACT_M * z
    pinch[:3, 3] = contact
    return grasp, pinch


@pytest.mark.parametrize('arm', ARMS)
def test_g1_frames_from_golem_upper_ik_are_the_dex3_frames_the_skills_had(arm):
    """golem_upper_ik's G1 grasp frame and the convention frame on it keep the old poses."""
    from golem_upper_ik.profile import load_profile
    g1 = robot_profile('g1')
    upper = dict(zip(ARMS, load_profile('g1').grasp_frames))[arm]
    assert upper.parent == f'{arm}_hand_palm_link'
    old_grasp, old_pinch = _old_dex3_frames(arm)
    np.testing.assert_allclose(upper.offset, old_pinch, atol=1e-12)
    tool = g1.tool_frames[g1.grasp_frames[arm]]
    assert tool.parent == g1.pinch_frames[arm] == upper.name
    np.testing.assert_allclose(upper.offset @ tool.matrix(), old_grasp, atol=1e-12)


def test_g1_profile():
    """The G1 grasps with the Dex3-1, from its prep posture, with neither GraspGenX nor a servo."""
    g1 = robot_profile('g1')
    assert (g1.hand, g1.graspgen_gripper, g1.hand_cameras) == ('dex3', None, False)
    assert all(posture == g1.prep_posture for posture in g1.staging_postures.values())
    assert 'graspgenx' in _refusal(g1, 'graspgenx')
    assert _refusal(g1, 'heuristic') is None
