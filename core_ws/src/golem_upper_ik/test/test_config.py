'''
The controller-config loader, on the carried configs and on configs written by the tests.

test/fixtures/old_controller/<robot>.json holds what the robot's previous arm controller's
load_controller_config returned for the same configs (test/fixtures/README.md); these tests
import no old package.
'''
import json
import math
from pathlib import Path
import re

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest
import yaml

FIXTURES = Path(__file__).parent / 'fixtures' / 'old_controller'
CONFIGS = {
    'h12': ['safety_split', 'sim_safety_split', 'safety_full', 'debug'],
    'g1': ['g1_split', 'sim_split'],
}
CASES = [f'{robot}/{name}' for robot, names in CONFIGS.items() for name in names]
HELD = {
    'h12': ('torso_joint',),
    'g1': ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'),
}
# the keys both old loaders return under controller, with the same meaning
SHARED_CONTROLLER_KEYS = (
    'v_lim', 'w_lim', 'dq_lim', 'd_min', 'timeout',
    'threshold_ik', 'threshold_joint', 'threshold_linear', 'threshold_angular',
)
G1_CONTROLLER_KEYS = (
    'state_timeout', 'startup_timeout', 'startup_move_timeout', 'follow_tolerance', 'imu_frame',
)


def needs_robot(robot):
    if robot == 'g1':
        pytest.importorskip('g1_description')


def load(robot, name_or_path):
    needs_robot(robot)
    from golem_upper_ik.config import load_controller_config
    return load_controller_config(robot, name_or_path)


def old_config(robot, name):
    return json.loads((FIXTURES / f'{robot}.json').read_text())['configs'][name]


def installed(robot, name):
    share = Path(get_package_share_directory('golem_upper_ik'))
    return share / 'config' / robot / f'{name}.yaml'


def ros_name(dds_topic):
    assert dds_topic.startswith('rt/'), dds_topic
    return '/' + dds_topic[len('rt/'):]


def write_config(tmp_path, robot, raw, directory=None):
    '''raw written as YAML to tmp_path/<directory or robot>/test.yaml'''
    path = tmp_path / (directory or robot) / 'test.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def edited(robot, name, **sections):
    '''The installed config name of robot with each keyword section's keys replaced'''
    raw = yaml.safe_load(installed(robot, name).read_text())
    for section, values in sections.items():
        raw[section] = dict(raw.get(section) or {}, **values)
    return raw


def minimal(robot):
    '''The gains alone: every other key takes its default'''
    return {'gains': {'kp': 10.0, 'kd': 1.0, 'ki': 0.0}}


@pytest.mark.parametrize('case', CASES)
def test_configs_load(case):
    robot, name = case.split('/')
    config = load(robot, name)
    old = old_config(robot, name)['controller']

    assert config['robot'] == robot
    # the file names every held joint, so the values below are carried, not the defaults
    raw = yaml.safe_load(installed(robot, name).read_text())
    assert sorted(raw['controller']['hold_targets']) == sorted(HELD[robot])
    hold = config['controller']['hold_targets']
    if robot == 'h12':
        assert hold == {'torso_joint': old['torso_target']}
    else:
        assert hold == dict(zip(HELD['g1'], old['waist_target']))
    assert list(hold) == list(HELD[robot])


@pytest.mark.parametrize('case', CASES)
def test_config_values_match_old_loaders(case):
    robot, name = case.split('/')
    config = load(robot, name)
    old = old_config(robot, name)

    for key in ('low_cmd', 'low_state'):
        assert config['topics'][key] == ros_name(old['topics'][key]), key
    for key in SHARED_CONTROLLER_KEYS:
        assert config['controller'][key] == old['controller'][key], key
    for key in ('ctrl_hz', 'pub_hz'):
        assert config['frequency'][key] == old['frequency'][key], key
    for key in ('kp', 'kd', 'ki'):
        gains = config['gains'][key]
        assert gains.dtype == np.float32, key
        np.testing.assert_array_equal(gains, old['gains'][key], err_msg=key)
    clip = config['limits']
    np.testing.assert_array_equal(clip['q_clip_limits'][:, 0], old['clip']['q_low'])
    np.testing.assert_array_equal(clip['q_clip_limits'][:, 1], old['clip']['q_high'])
    np.testing.assert_array_equal(clip['dq_clip_limits'], old['clip']['dq'])
    np.testing.assert_array_equal(clip['tau_clip_limits'], old['clip']['tau'])
    # the old loaders' logging section configured a command recorder no node started
    assert 'logging' not in config

    # the planner section is the file's, with enabled and frame_names filled in
    planner = dict(config['planner'])
    assert planner.pop('enabled') == old['planner'].get('enabled', True)
    frame_names = planner.pop('frame_names')
    assert frame_names == old['planner'].get('frame_names', frame_names)
    assert planner == {key: value for key, value in old['planner'].items()
                       if key not in ('enabled', 'frame_names')}

    if robot == 'h12':
        # keys the H1-2 loader had and the G1's did not
        assert config['controller']['max_state_lag_ms'] == old['controller']['max_state_lag_ms']
        assert config['topics']['safety_out'] == '/lowcmd'
        assert config['controller']['imu_frame'] == 'torso_link'
        assert frame_names == ['left_grasp_frame', 'right_grasp_frame']
    else:
        # keys the G1 loader had and the H1-2's did not
        assert config['topics']['safety_out'] == ros_name(old['topics']['safety_out'])
        for key in G1_CONTROLLER_KEYS:
            assert config['controller'][key] == old['controller'][key], key
        assert config['frequency']['check_hz'] == old['frequency']['check_hz']
        assert config['controller']['max_state_lag_ms'] == 0.0


def test_every_installed_config_has_a_fixture():
    share = Path(get_package_share_directory('golem_upper_ik')) / 'config'
    for robot, names in CONFIGS.items():
        assert sorted(path.stem for path in (share / robot).glob('*.yaml')) == sorted(names)
        fixture = json.loads((FIXTURES / f'{robot}.json').read_text())
        assert sorted(fixture['configs']) == sorted(names)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_defaults(robot, tmp_path):
    '''A config with only gains: the loaders' shared defaults and each robot's own'''
    needs_robot(robot)
    from golem_upper_ik.profile import load_profile
    profile = load_profile(robot)
    config = load(robot, write_config(tmp_path, robot, minimal(robot)))

    assert config['topics'] == {
        'low_cmd': '/safety/lowcmd_upper_in',
        'low_state': '/lowstate',
        'safety_out': '/lowcmd',
    }
    controller = config['controller']
    assert controller['hold_targets'] == {joint: 0.0 for joint in HELD[robot]}
    assert controller['max_state_lag_ms'] == (50.0 if robot == 'h12' else 0.0)
    assert controller['imu_frame'] == profile.imu_frame
    expected = {
        'v_lim': 1.0, 'w_lim': 2.0, 'dq_lim': 1.0, 'd_min': 0.02, 'timeout': 10.0,
        'threshold_ik': 0.001, 'threshold_joint': 0.01, 'threshold_linear': 0.002,
        'threshold_angular': 0.01, 'state_timeout': 5.0, 'startup_timeout': 30.0,
        'startup_move_timeout': 30.0, 'follow_tolerance': 0.05,
    }
    for key, value in expected.items():
        assert controller[key] == value, key
    assert config['frequency'] == {'ctrl_hz': 50.0, 'pub_hz': 500.0, 'check_hz': 1000.0}
    assert config['planner'] == {'enabled': True, 'frame_names': list(profile.planner_frames)}

    # clip limits from the profile's limits, with offset 0.001, ratios 0.10 and 0.25
    limits = config['limits']
    np.testing.assert_array_equal(limits['q_clip_limits'][:, 0], profile.q_low + 0.001)
    np.testing.assert_array_equal(limits['q_clip_limits'][:, 1], profile.q_high - 0.001)
    np.testing.assert_array_equal(limits['dq_clip_limits'], profile.dq_max * 0.10)
    np.testing.assert_array_equal(limits['tau_clip_limits'], profile.tau_max * 0.25)

    # scalar gains on the upper slots, 0 on the legs
    kp = config['gains']['kp']
    assert kp.shape == (len(profile.body_joints),)
    assert list(np.flatnonzero(kp)) == list(profile.upper_slots)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_joint_lists_take_body_or_upper_lengths(robot, tmp_path):
    needs_robot(robot)
    from golem_upper_ik.profile import load_profile
    profile = load_profile(robot)
    body, upper = len(profile.body_joints), len(profile.upper_slots)
    raw = minimal(robot)
    raw['gains']['kd'] = [float(i) for i in range(upper)]
    raw['limits'] = {'clip': {'velocity_ratio': [0.5] * body,
                              'torque_ratio': [0.2] * upper}}
    config = load(robot, write_config(tmp_path, robot, raw))

    kd = config['gains']['kd']
    np.testing.assert_array_equal(kd[list(profile.upper_slots)], range(upper))
    np.testing.assert_array_equal(kd[list(profile.lower_slots)], 0.0)
    np.testing.assert_array_equal(config['limits']['dq_clip_limits'], profile.dq_max * 0.5)
    tau = config['limits']['tau_clip_limits']
    np.testing.assert_array_equal(tau[list(profile.lower_slots)], 0.0)
    np.testing.assert_array_equal(tau[list(profile.upper_slots)],
                                  profile.tau_max[list(profile.upper_slots)] * 0.2)

    raw['gains']['kd'] = [1.0] * (upper - 1)
    with pytest.raises(ValueError, match='gains.kd'):
        load(robot, write_config(tmp_path, robot, raw))


def test_low_cmd_topic_must_be_safety_input(tmp_path):
    for low_cmd in ('/lowcmd', 'rt/safety/lowcmd_upper_in', 'safety/lowcmd_in'):
        raw = edited('h12', 'safety_split', topics={'low_cmd': low_cmd})
        with pytest.raises(ValueError, match='topics.low_cmd'):
            load('h12', write_config(tmp_path, 'h12', raw))
    raw = edited('h12', 'safety_split', topics={'low_cmd': '/safety/lowcmd_in'})
    assert load('h12', write_config(tmp_path, 'h12', raw))['topics']['low_cmd'] == (
        '/safety/lowcmd_in')


def test_topics_are_ros_names(tmp_path):
    raw = edited('h12', 'safety_split', topics={'low_state': 'rt/lowstate'})
    with pytest.raises(ValueError, match="topics.low_state.*'/lowstate'"):
        load('h12', write_config(tmp_path, 'h12', raw))


def test_safety_out_is_an_output_of_the_robot(tmp_path):
    raw = edited('h12', 'safety_split', topics={'safety_out': '/safety/lowcmd_upper_in'})
    with pytest.raises(ValueError, match='topics.safety_out'):
        load('h12', write_config(tmp_path, 'h12', raw))
    raw = edited('g1', 'g1_split', topics={'safety_out': '/lowcmd'})
    assert load('g1', write_config(tmp_path, 'g1', raw))['topics']['safety_out'] == '/lowcmd'
    raw = edited('g1', 'g1_split', topics={'safety_out': '/safety/lowcmd_upper_in'})
    with pytest.raises(ValueError, match='topics.safety_out'):
        load('g1', write_config(tmp_path, 'g1', raw))


@pytest.mark.parametrize('robot, name', [('h12', 'safety_split'), ('g1', 'g1_split')])
def test_arm_sdk_output_refused(robot, name, tmp_path):
    '''golem_safety writes /lowcmd only: a follow gate on another topic is refused, naming it'''
    raw = edited(robot, name, topics={'safety_out': '/arm_sdk'})
    with pytest.raises(ValueError, match=re.escape(
            f"key 'topics.safety_out': '/arm_sdk' is not a golem_safety output of robot "
            f'{robot} (/lowcmd)')):
        load(robot, write_config(tmp_path, robot, raw))


@pytest.mark.parametrize('key, raw_edit, message', [
    ('bogus', {'bogus': 1}, 'unknown'),
    ('controller.bogus', {'controller': {'bogus': 1}}, 'unknown'),
    ('planner.bogus', {'planner': {'bogus': 1}}, 'unknown'),
    ('limits.clip.bogus', {'limits': {'clip': {'bogus': 1}}}, 'unknown'),
    ('zmp', {'zmp': {}}, 'unknown'),
    ('logging', {'logging': {'enabled': False}}, 'unknown'),
    ('mode', {'mode': 'debug'}, 'removed'),
    ('network', {'network': {'domain_id': 1}}, 'ROS supplies the domain'),
    ('controller.torso_target', {'controller': {'torso_target': 0.0}}, 'hold_targets'),
    ('controller.waist_target', {'controller': {'waist_target': 0.0}}, 'hold_targets'),
    ('limits.estop', {'limits': {'estop': {'velocity_ratio': 0.5}}}, 'no estop of its own'),
    ('frequency.bogus', {'frequency': {'bogus': 1.0}}, 'unknown'),
    ('planner.frame_z_min_margin', {'planner': {'frame_z_min_margin': 0.01}},
     'frame_limit_margin'),
])
def test_unknown_and_replaced_keys_raise(tmp_path, key, raw_edit, message):
    raw = minimal('h12')
    for section, value in raw_edit.items():
        raw[section] = value
    with pytest.raises(ValueError, match=f"'{key}'.*{message}"):
        load('h12', write_config(tmp_path, 'h12', raw))


def test_hold_targets_name_held_joints(tmp_path):
    raw = minimal('h12')
    raw['controller'] = {'hold_targets': {'waist_yaw_joint': 0.0}}
    with pytest.raises(ValueError, match='waist_yaw_joint.*torso_joint'):
        load('h12', write_config(tmp_path, 'h12', raw))

    raw = minimal('g1')
    raw['controller'] = {'hold_targets': {'waist_roll_joint': 0.1}}
    hold = load('g1', write_config(tmp_path, 'g1', raw))['controller']['hold_targets']
    assert hold == {'waist_yaw_joint': 0.0, 'waist_roll_joint': 0.1, 'waist_pitch_joint': 0.0}

    raw['controller'] = {'hold_targets': 0.1}
    with pytest.raises(ValueError, match='hold_targets'):
        load('g1', write_config(tmp_path, 'g1', raw))


@pytest.mark.parametrize('section, key, value', [
    ('controller', 'v_lim', float('nan')),
    ('controller', 'timeout', float('inf')),
    ('controller', 'state_timeout', 0.0),
    ('controller', 'follow_tolerance', -0.01),
    ('frequency', 'ctrl_hz', 0.0),
    ('frequency', 'check_hz', -1.0),
    ('controller', 'v_lim', True),
    ('controller', 'v_lim', 'fast'),
])
def test_bad_values_raise(tmp_path, section, key, value):
    raw = minimal('h12')
    raw[section] = {key: value}
    with pytest.raises(ValueError, match=f'{section}.{key}'):
        load('h12', write_config(tmp_path, 'h12', raw))


# a speed limit, timeout, convergence threshold or interval must be positive; a distance,
# ratio or offset must not be negative (a ratio of 0 is the lower slots' value in an
# upper-body list)
@pytest.mark.parametrize('key, value', [
    ('controller.v_lim', 0.0),
    ('controller.w_lim', -1.0),
    ('controller.dq_lim', 0.0),
    ('controller.timeout', 0.0),
    ('controller.threshold_ik', 0.0),
    ('controller.threshold_joint', 0.0),
    ('controller.threshold_linear', -0.001),
    ('controller.threshold_angular', 0.0),
    ('controller.d_min', -0.01),
    ('limits.clip.position_offset', -0.001),
    ('limits.clip.velocity_ratio', -0.1),
    ('limits.clip.torque_ratio', -0.25),
    ('limits.clip.torque_ratio', [0.25] * 14 + [-0.25]),
])
def test_nonsensical_values_raise(tmp_path, key, value):
    raw = minimal('h12')
    *sections, name = key.split('.')
    target = raw
    for section in sections:
        target = target.setdefault(section, {})
    target[name] = value
    with pytest.raises(ValueError, match=re.escape(f"'{key}'")):
        load('h12', write_config(tmp_path, 'h12', raw))


@pytest.mark.parametrize('key, value', [
    ('try_direct', 'false'),
    ('timeout', float('nan')),
    ('frame_names', 'left_grasp_frame'),
    ('frame_names', ['left_grasp_frame']),
    ('frame_names', ['left_grasp_frame', '']),
    ('max_interpolation_steps', 2.5),
    ('min_interpolation_steps', 0),
    ('constraint_check_steps', True),
    ('planner', 5),
    ('moving_speed', 0.0),
    ('frame_x_max', 'far'),
    ('frame_limit_margin', -0.01),
    ('enabled', 'yes'),
])
def test_bad_planner_values_raise(tmp_path, key, value):
    raw = minimal('h12')
    raw['planner'] = {key: value}
    with pytest.raises(ValueError, match=re.escape(f"'planner.{key}'")):
        load('h12', write_config(tmp_path, 'h12', raw))


def test_planner_values_load_with_their_types(tmp_path):
    planner = load('h12', 'safety_split')['planner']
    for key in ('enabled', 'try_direct', 'simplify', 'shortcut'):
        assert type(planner[key]) is bool, key
    for key in ('min_interpolation_steps', 'max_interpolation_steps', 'constraint_check_steps'):
        assert type(planner[key]) is int, key
    for key in ('timeout', 'range', 'simplify_time', 'moving_speed', 'validity_resolution',
                'frame_x_max', 'frame_y_min', 'frame_y_max', 'frame_z_min', 'frame_z_max',
                'frame_limit_margin', 'frame_z_corridor_margin', 'start_limit_tolerance'):
        assert type(planner[key]) is float, key
    assert planner['planner'] == 'RRTConnect'
    assert planner['frame_names'] == ['left_grasp_frame', 'right_grasp_frame']

    # a frame bound may be null (no bound); a whole-number count may be written as a float
    raw = minimal('h12')
    raw['planner'] = {'frame_y_min': None, 'max_interpolation_steps': 30.0}
    planner = load('h12', write_config(tmp_path, 'h12', raw))['planner']
    assert planner['frame_y_min'] is None
    assert planner['max_interpolation_steps'] == 30
    assert type(planner['max_interpolation_steps']) is int


def test_mode_without_an_output_topic_raises(monkeypatch):
    import dataclasses
    from golem_upper_ik import config as config_module
    from golem_upper_ik.profile import load_profile
    profile = dataclasses.replace(load_profile('h12'), modes=('full_body_mode', 'wobble_mode'))
    monkeypatch.setattr(config_module, 'load_profile', lambda robot: profile)
    with pytest.raises(ValueError, match='h12.*wobble_mode'):
        config_module.load_controller_config('h12', 'safety_split')


def test_numbers_written_without_a_dot_load(tmp_path):
    # PyYAML reads 1e-3 (no dot) as a string; the old loaders' float() took it as a number
    path = tmp_path / 'h12' / 'test.yaml'
    path.parent.mkdir(parents=True)
    path.write_text('gains: {kp: 10.0, kd: 1.0, ki: 0.0}\ncontroller: {threshold_ik: 1e-3}\n')
    assert load('h12', path)['controller']['threshold_ik'] == 0.001


def test_two_spellings_of_one_key_raise(tmp_path):
    raw = minimal('h12')
    raw['limits'] = {'clip': {'velocity_ratio': 0.1}, 'enforce': {'velocity_ratio': 0.2}}
    with pytest.raises(ValueError, match='limits.enforce'):
        load('h12', write_config(tmp_path, 'h12', raw))
    raw = minimal('h12')
    raw['limits'] = {'enforce': {'velocity_ratio': 0.2}}
    config = load('h12', write_config(tmp_path, 'h12', raw))
    from golem_upper_ik.profile import load_profile
    np.testing.assert_array_equal(config['limits']['dq_clip_limits'],
                                  load_profile('h12').dq_max * 0.2)


def test_position_offset_too_large_raises(tmp_path):
    raw = minimal('h12')
    raw['limits'] = {'clip': {'position_offset': 0.5}}
    with pytest.raises(ValueError, match='position_offset'):
        load('h12', write_config(tmp_path, 'h12', raw))


def test_names_resolve_under_the_robots_config_directory():
    with_suffix = load('h12', 'safety_split.yaml')
    bare = load('h12', 'safety_split')
    assert with_suffix['source'] == bare['source'] == str(installed('h12', 'safety_split'))
    with pytest.raises(FileNotFoundError, match='g1_split'):
        load('h12', 'g1_split')


def test_config_of_other_robot_is_refused(tmp_path):
    with pytest.raises(ValueError, match='robot g1'):
        load('h12', str(installed('g1', 'g1_split')))
    with pytest.raises(ValueError, match='robot h12'):
        load('g1', write_config(tmp_path, 'g1', minimal('g1'), directory='h12'))


def test_unknown_robot_raises():
    from golem_upper_ik.config import load_controller_config
    with pytest.raises(ValueError, match='h1'):
        load_controller_config('h1', 'safety_split')


def test_values_are_finite_after_loading():
    for robot, names in CONFIGS.items():
        for name in names:
            config = load(robot, name)
            for section in ('controller', 'frequency'):
                for key, value in config[section].items():
                    if isinstance(value, float):
                        assert math.isfinite(value), (robot, name, section, key)


@pytest.mark.parametrize('name', CONFIGS['g1'])
def test_g1_waist_gains_are_damped_from_the_urdf_inertia(models, name):
    import pinocchio as pin
    gains = load('g1', name)['gains']
    model = models('g1')
    profile = model.profile
    # joint-space inertia at home (arms and waist at 0)
    data = model.model_body.createData()
    inertia = np.diag(pin.crba(model.model_body, data, np.zeros(model.num_motor)))
    for joint in profile.held_joints:
        slot = profile.body_joints.index(joint)
        zeta = gains['kd'][slot] / (2.0 * np.sqrt(gains['kp'][slot] * inertia[slot]))
        assert 0.4 <= zeta <= 0.5, (name, joint, zeta)
    arms = [profile.body_joints.index(joint) for joint in profile.enabled_joints]
    assert np.all(gains['kd'][arms] == 1.5)
