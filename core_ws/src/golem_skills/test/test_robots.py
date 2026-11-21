"""Test the per-robot profiles the manipulation skills read (robots.py)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from golem_skills.robots import ARMS, DEFAULT_ROBOT, ROBOTS, robot_profile


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
