'''
The robot profiles against golem_safety's robot data.
'''
import dataclasses
import json
import os
from pathlib import Path
import types

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest
import yaml
from golem_upper_ik.profile import known_robots

ROBOTS = known_robots()

EXPECTED = {
    'h12': {
        'joints': 27,
        'mode_machine': 6,
        'lower_slots': range(0, 12),
        'upper_slots': range(12, 27),
        'ee_frames': ('left_wrist_yaw_link', 'right_wrist_yaw_link'),
        'planner_frames': ('left_grasp_frame', 'right_grasp_frame'),
        'imu_frame': 'torso_link',
        'held_joints': ('torso_joint',),
        'urdf': 'h1_2_magpie.urdf',
        'urdf_ros': 'h1_2_magpie_ros.urdf',
        'collision': ('h1_2_magpie_sphere.urdf', 'h1_2_magpie_sphere_collision.srdf'),
        'shutdown_keeps_held_joints': False,
    },
    'g1': {
        'joints': 29,
        'mode_machine': 5,
        'lower_slots': range(0, 12),
        'upper_slots': range(12, 29),
        'ee_frames': ('left_hand_palm_link', 'right_hand_palm_link'),
        'planner_frames': ('left_hand_palm_link', 'right_hand_palm_link'),
        'imu_frame': 'imu_in_pelvis',
        'held_joints': ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'),
        'urdf': 'g1_29dof_with_hand_rev_1_0.urdf',
        'urdf_ros': 'g1_29dof_with_hand_rev_1_0_ros.urdf',
        'collision': ('g1_sphere.urdf', 'g1_sphere_collision.srdf'),
        'shutdown_keeps_held_joints': True,
    },
}


def profile_of(robot):
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def safety_data(robot):
    share = Path(get_package_share_directory('golem_safety'))
    return yaml.safe_load((share / 'robots' / f'{robot}.yaml').read_text())


@pytest.mark.parametrize('robot', ROBOTS)
def test_profiles_load(robot):
    profile = profile_of(robot)
    expected = EXPECTED[robot]

    assert profile.robot == robot
    assert len(profile.body_joints) == expected['joints']
    assert profile.mode_machine == expected['mode_machine']
    assert profile.mode_pr == 0
    assert profile.lower_slots == expected['lower_slots']
    assert profile.upper_slots == expected['upper_slots']
    assert profile.ee_frames == expected['ee_frames']
    assert profile.planner_frames == expected['planner_frames']
    assert profile.imu_frame == expected['imu_frame']
    assert profile.held_joints == expected['held_joints']
    assert (profile.shutdown_keeps_held_joints
            is expected['shutdown_keeps_held_joints'])
    for name in ('q_low', 'q_high', 'dq_max', 'tau_max'):
        values = getattr(profile, name)
        assert values.shape == (expected['joints'],), name
        assert values.dtype == np.float64, name
        assert not values.flags.writeable, name

    assert Path(profile.urdf_path).name == expected['urdf']
    assert Path(profile.urdf_ros_path).name == expected['urdf_ros']
    assert os.path.isfile(profile.urdf_path), profile.urdf_path
    assert os.path.isfile(profile.urdf_ros_path), profile.urdf_ros_path
    assert tuple(Path(path).name for path in profile.collision) == expected['collision']
    for path in profile.collision:
        assert os.path.isfile(path), path

    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.robot = 'other'
    assert isinstance(profile.named_configs, types.MappingProxyType)
    with pytest.raises(TypeError):
        profile.named_configs['home'] = np.zeros(len(profile.enabled_joints))
    # compared and hashed by identity: an array field has no truth value to compare by
    assert profile == profile
    assert (profile == profile_of(robot)) is False
    assert hash(profile) == hash(profile)


@pytest.mark.parametrize('fault', ['gap', 'repeat'])
def test_joint_slots_must_be_0_to_n(monkeypatch, fault):
    from golem_upper_ik import profile as profile_module
    data = profile_module._robot_data('h12')
    joints = [dict(joint) for joint in data['joints']]
    joints[-1]['slot'] = len(joints) if fault == 'gap' else 0
    monkeypatch.setattr(profile_module, '_robot_data', lambda robot: dict(data, joints=joints))
    with pytest.raises(ValueError, match='slots'):
        profile_module.load_profile('h12')


@pytest.mark.parametrize('robot', ROBOTS)
def test_profile_body_joints_match_safety_data(robot):
    profile = profile_of(robot)
    data = safety_data(robot)
    joints = data['joints']

    assert profile.body_joints == tuple(joint['name'] for joint in joints)
    assert profile.mode_machine == data['mode_machine']
    assert profile.mode_pr == data['mode_pr']
    assert profile.modes == tuple(data['modes'])
    np.testing.assert_array_equal(profile.q_low, [joint['position'][0] for joint in joints])
    np.testing.assert_array_equal(profile.q_high, [joint['position'][1] for joint in joints])
    np.testing.assert_array_equal(profile.dq_max, [joint['velocity'] for joint in joints])
    np.testing.assert_array_equal(profile.tau_max, [joint['effort'] for joint in joints])
    first, last = data['lower_slots']
    assert profile.lower_slots == range(first, last + 1)
    first, last = data['upper_slots']
    assert profile.upper_slots == range(first, last + 1)


@pytest.mark.parametrize('robot', ROBOTS)
def test_held_joints_are_the_upper_joints_the_ik_leaves(robot):
    profile = profile_of(robot)
    upper = [profile.body_joints[i] for i in profile.upper_slots]
    assert profile.held_joints == tuple(
        joint for joint in upper if joint not in profile.enabled_joints)
    assert set(profile.enabled_joints) <= set(upper)


@pytest.mark.parametrize('robot', ROBOTS)
def test_named_configs_are_over_the_ik_joints(robot):
    profile = profile_of(robot)
    for name, value in profile.named_configs.items():
        assert value.shape == (len(profile.enabled_joints),), name
        assert value.dtype == np.float64, name
    assert set(profile.startup_sequence) <= set(profile.named_configs)


def test_unknown_robot_raises():
    from golem_upper_ik.profile import load_profile, robot_config
    for function in (load_profile, robot_config):
        with pytest.raises(ValueError, match="unknown robot 'h1'.*g1, h12"):
            function('h1')


def test_known_robots():
    assert known_robots() == ('g1', 'h12')


def test_a_robot_needs_both_packages_data_to_be_known(monkeypatch, tmp_path):
    from golem_upper_ik import profile as profile_module
    (tmp_path / 'golem_safety' / 'robots').mkdir(parents=True)
    (tmp_path / 'golem_upper_ik' / 'config' / 'a').mkdir(parents=True)
    (tmp_path / 'golem_upper_ik' / 'config' / 'b').mkdir(parents=True)
    (tmp_path / 'golem_safety' / 'robots' / 'a.yaml').write_text('robot: a\n')
    (tmp_path / 'golem_safety' / 'robots' / 'c.yaml').write_text('robot: c\n')
    (tmp_path / 'golem_upper_ik' / 'config' / 'a' / 'robot.yaml').write_text('{}\n')
    (tmp_path / 'golem_upper_ik' / 'config' / 'b' / 'robot.yaml').write_text('{}\n')
    monkeypatch.setattr(profile_module, 'get_package_share_directory',
                        lambda package: str(tmp_path / package))
    assert profile_module.known_robots() == ('a',)


# robot_config is compared with the expected values of the JSON file this variable names
# ({'robots': {robot: robot_config(robot)}}); the file is not in the repository, so the test
# is skipped without it
REFERENCE_VARIABLE = 'GOLEM_ROBOT_CONFIG_REFERENCE'


@pytest.mark.parametrize('robot', ROBOTS)
def test_robot_config_matches_reference(robot):
    from golem_upper_ik.profile import robot_config
    path = os.environ.get(REFERENCE_VARIABLE)
    if not path:
        pytest.skip(f'{REFERENCE_VARIABLE} names no reference file')
    with open(path, encoding='utf-8') as handle:
        reference = json.load(handle)['robots'][robot]
    config = robot_config(robot)
    assert sorted(config) == sorted(reference)
    for key, value in reference.items():
        assert config[key] == value, key


def robot_config_with(monkeypatch, tmp_path, robot, change):
    '''The profile module reading a copy of robot's robot.yaml that change edits'''
    from golem_upper_ik import profile as profile_module
    real = get_package_share_directory('golem_upper_ik')
    raw = yaml.safe_load((Path(real) / 'config' / robot / 'robot.yaml').read_text())
    change(raw)
    path = tmp_path / 'golem_upper_ik' / 'config' / robot / 'robot.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw))

    def share(package):
        return str(tmp_path / package) if package == 'golem_upper_ik' else (
            get_package_share_directory(package))
    monkeypatch.setattr(profile_module, 'get_package_share_directory', share)
    return profile_module.robot_config(robot)


@pytest.mark.parametrize('robot', ROBOTS)
def test_robot_config_rejects_unknown_and_missing_keys(monkeypatch, tmp_path, robot):
    with pytest.raises(ValueError, match=f"{robot}.*'extra_key'"):
        robot_config_with(monkeypatch, tmp_path / 'extra', robot,
                          lambda raw: raw.update(extra_key=1))
    with pytest.raises(ValueError, match=f"{robot}.*'imu_frame'"):
        robot_config_with(monkeypatch, tmp_path / 'missing', robot,
                          lambda raw: raw.pop('imu_frame'))


def test_robot_config_rejects_a_bad_value(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="'named_configs'.*home"):
        robot_config_with(monkeypatch, tmp_path, 'h12',
                          lambda raw: raw['named_configs'].update(home='zero'))
