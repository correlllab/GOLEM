'''
The robot profiles against golem_safety's robot data and the old controllers' tables.

test/fixtures/old_controller/<robot>.json holds the previous arm controllers' tables
(test/fixtures/README.md); these tests import no old package.
'''
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import types

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest
import yaml

FIXTURES = Path(__file__).parent / 'fixtures' / 'old_controller'
ROBOTS = ('h12', 'g1')

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
        'shutdown_keeps_held_joints_on_lowcmd': False,
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
        'collision': None,
        'shutdown_keeps_held_joints_on_lowcmd': True,
    },
}


def profile_of(robot):
    if robot == 'g1':
        pytest.importorskip('g1_description')
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def old_tables(robot):
    return json.loads((FIXTURES / f'{robot}.json').read_text())


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
    assert (profile.shutdown_keeps_held_joints_on_lowcmd
            is expected['shutdown_keeps_held_joints_on_lowcmd'])
    for name in ('q_low', 'q_high', 'dq_max', 'tau_max'):
        values = getattr(profile, name)
        assert values.shape == (expected['joints'],), name
        assert values.dtype == np.float64, name
        assert not values.flags.writeable, name

    assert Path(profile.urdf_path).name == expected['urdf']
    assert Path(profile.urdf_ros_path).name == expected['urdf_ros']
    assert os.path.isfile(profile.urdf_path), profile.urdf_path
    assert os.path.isfile(profile.urdf_ros_path), profile.urdf_ros_path
    if expected['collision'] is None:
        assert profile.collision is None
    else:
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
def test_profile_matches_old_controller_tables(robot):
    profile = profile_of(robot)
    old = old_tables(robot)

    assert profile.body_joints == tuple(old['body_joints'])
    assert list(profile.upper_slots) == old['upper_body_index']
    assert list(profile.lower_slots) == old['lower_body_index']
    assert [profile.body_joints[i] for i in profile.upper_slots] == old['upper_body_joints']
    assert profile.enabled_joints == tuple(old['enabled_joints'])
    # the held joints are the upper-body joints the IK does not solve for
    assert profile.held_joints == tuple(
        joint for joint in old['upper_body_joints'] if joint not in old['enabled_joints'])
    assert profile.planner_frames == tuple(old['planner_frame_names'])
    if 'ee_frames' in old:
        assert profile.ee_frames == tuple(old['ee_frames'])

    limits = old['joint_limits']
    np.testing.assert_array_equal(profile.q_low, [low for low, _ in limits['position']])
    np.testing.assert_array_equal(profile.q_high, [high for _, high in limits['position']])
    np.testing.assert_array_equal(profile.dq_max, limits['velocity'])
    np.testing.assert_array_equal(profile.tau_max, limits['effort'])


@pytest.mark.parametrize('robot', ROBOTS)
def test_named_configs_match_old_controller(robot):
    profile = profile_of(robot)
    old = old_tables(robot)

    assert list(profile.named_configs) == list(old['named_configs'])
    for name, q in old['named_configs'].items():
        value = profile.named_configs[name]
        assert value.shape == (len(profile.enabled_joints),), name
        assert value.dtype == np.float64, name
        np.testing.assert_array_equal(value, q, err_msg=name)
    assert profile.startup_sequence == tuple(old['startup_sequence'])


def test_unknown_robot_raises():
    from golem_upper_ik.profile import load_profile
    with pytest.raises(ValueError, match='h1'):
        load_profile('h1')


# Run in a fresh interpreter, where an import hook refuses g1_description: the H1-2's
# profile, configs, pairing and nodes must work without it, and the G1 module must import
# without it (only load_profile('g1') reaches for it). The last step checks that the hook
# works.
NO_G1_SCRIPT = r'''
import importlib.abc
import sys

attempts = []


class RefuseG1Description(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'g1_description' or name.startswith('g1_description.'):
            attempts.append(name)
            raise ImportError(f'{name} is not installed')
        return None


sys.meta_path.insert(0, RefuseG1Description())
assert 'g1_description' not in sys.modules

import golem_upper_ik.robots.h12
import golem_upper_ik.robots.g1
import golem_upper_ik.ros2.frame_task_server
import golem_upper_ik.ros2.joint_state_publisher
from golem_upper_ik.config import load_controller_config
from golem_upper_ik.pairing import default_controller_config, pairing_refusal
from golem_upper_ik.profile import load_profile

profile = load_profile('h12')
for name in ('safety_split', 'sim_safety_split', 'safety_full', 'debug'):
    load_controller_config('h12', name)
assert pairing_refusal('h12', 'safety_split', 'relax_safety_split') is None
assert pairing_refusal('g1', 'g1_split', 'split') is None
assert default_controller_config('h12', 'relax_safety_split') == 'safety_split'
assert attempts == [], attempts
assert not any(m.split('.')[0] == 'g1_description' for m in sys.modules)

try:
    load_profile('g1')
except ImportError:
    pass
else:
    raise AssertionError('load_profile("g1") loaded although g1_description was refused')
assert attempts, 'load_profile("g1") never imported g1_description'
print('ok')
'''


def test_h12_profile_needs_no_g1_package(tmp_path):
    done = subprocess.run([sys.executable, '-c', NO_G1_SCRIPT], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().splitlines()[-1] == 'ok'
