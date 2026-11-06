"""Test the robot profile: golem_safety's joint table and this package's robot.yaml."""

import numpy as np
import pytest

from golem_lowerbody_rl.profile import known_robots, load_profile


def test_both_robots_are_known():
    """Each robot with a joint table and a robot.yaml is known."""
    assert set(known_robots()) >= {'h12', 'g1'}


@pytest.mark.parametrize('robot, first_leg', [
    ('h12', 'left_hip_yaw_joint'),
    ('g1', 'left_hip_pitch_joint'),
])
def test_slots_follow_the_safety_table(robot, first_leg):
    """A joint's slot is its index in golem_safety's table, and the legs are slots 0-11."""
    profile = load_profile(robot)
    assert profile.robot == robot
    assert profile.legs[0] == first_leg
    assert profile.slots([first_leg]).tolist() == [0]
    assert profile.slots(profile.legs).tolist() == list(range(12))
    assert profile.slots(['right_knee_joint']).tolist() == [9]
    assert profile.slots(profile.body_joints).tolist() == list(range(len(profile.body_joints)))
    assert profile.slots([]).shape == (0,)


def test_unknown_joint_is_refused():
    """slots() raises a KeyError naming a joint the robot does not have."""
    profile = load_profile('g1')
    with pytest.raises(KeyError, match='torso_joint'):
        profile.slots(['left_knee_joint', 'torso_joint'])


def test_mode_machine_from_the_safety_table():
    """mode_machine and mode_pr are golem_safety's values for each robot."""
    assert load_profile('h12').mode_machine == 6
    assert load_profile('g1').mode_machine == 5
    assert load_profile('h12').mode_pr == 0
    assert load_profile('g1').mode_pr == 0


@pytest.mark.parametrize('robot, policies, default', [
    ('h12', ('almi', 'walk', 'fame'), 'almi'),
    ('g1', ('homie',), 'homie'),
])
def test_robot_data(robot, policies, default):
    """Each robot's policies and default policy are its robot.yaml's."""
    profile = load_profile(robot)
    assert profile.policies == policies
    assert profile.default_policy == default
    assert len(profile.legs) == 12


def test_unknown_robot_is_refused():
    """A robot without data is refused, naming the known ones."""
    with pytest.raises(ValueError, match="unknown robot 'h13'"):
        load_profile('h13')


def _write(tmp_path, text):
    """Write a robot.yaml with text into tmp_path and return its path."""
    path = tmp_path / 'robot.yaml'
    path.write_text(text)
    return str(path)


G1_LEGS = ('[left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint, left_knee_joint, '
           'left_ankle_pitch_joint, left_ankle_roll_joint, right_hip_pitch_joint, '
           'right_hip_roll_joint, right_hip_yaw_joint, right_knee_joint, '
           'right_ankle_pitch_joint, right_ankle_roll_joint]')


def test_a_robot_yaml_path_loads(tmp_path):
    """load_profile reads a robot.yaml given by path in place of the installed one."""
    path = _write(tmp_path, f'legs: {G1_LEGS}\npolicies: [homie]\ndefault_policy: homie\n')
    assert load_profile('g1', path).legs == load_profile('g1').legs


@pytest.mark.parametrize('text, message', [
    (f'legs: {G1_LEGS}\npolicies: [homie]\n', "missing key 'default_policy'"),
    (f'legs: {G1_LEGS}\npolicies: [homie]\ndefault_policy: homie\ntrim: 0\n',
     "unknown key 'trim'"),
    (f'legs: {G1_LEGS}\npolicies: [homie]\ndefault_policy: walk\n',
     "default_policy 'walk' is not one of"),
    (f'legs: {G1_LEGS}\npolicies: [homie, homie]\ndefault_policy: homie\n',
     "'policies' must name each entry once"),
    (f'legs: {G1_LEGS.replace("left_knee_joint", "waist_yaw_joint")}\n'
     'policies: [homie]\ndefault_policy: homie\n', 'lower-body joints'),
    (f'legs: {G1_LEGS}\nimu_frame: pelvis\npolicies: [homie]\ndefault_policy: homie\n',
     "unknown key 'imu_frame'"),
    (f'legs: {G1_LEGS}\npolicies: [homie]\ndefault_policy: ""\n',
     "'default_policy' must be a name"),
], ids=['missing', 'unknown', 'default', 'repeated', 'not-legs', 'no-imu-frame',
        'empty-default'])
def test_bad_robot_yaml_is_refused(tmp_path, text, message):
    """A robot.yaml with a missing, unknown or inconsistent key is refused, naming it."""
    with pytest.raises(ValueError, match=message):
        load_profile('g1', _write(tmp_path, text))


def test_profile_is_frozen():
    """A profile cannot be changed after loading."""
    profile = load_profile('h12')
    with pytest.raises(AttributeError):
        profile.robot = 'g1'
    assert isinstance(profile.slots(profile.legs), np.ndarray)
