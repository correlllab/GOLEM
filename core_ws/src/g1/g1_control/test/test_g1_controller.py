'''
G1-specific checks: joint tables, model loading, IMU handling, postures,
configs, the follow gate, config pairing and the example tools' domain-0 guard
'''

import sys
import importlib.util
from pathlib import Path

import yaml
import numpy as np
import pinocchio as pin
import pytest
from pyquaternion import Quaternion

from g1_description.joints import BODY

from g1_control.core.robot_model import RobotModel
from g1_control.utility import controller_config
from g1_control.utility import joint_definition as jd
from g1_control.core.follow_gate import FollowGate, relays_commands
from g1_control.utility.named_config import NAMED_CONFIGS, STARTUP_SEQUENCE
from g1_control.utility.path_definition import URDF_BODY_PATH, URDF_PATH
from g1_control.utility.controller_config import load_controller_config
from g1_control.utility.config_pairing import (
    SAFETY_CONTROLLER_CONFIGS,
    UPPER_CHANNEL_SAFETY_CONFIGS,
    default_controller_config,
    pairing_refusal,
)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / 'config'
EXAMPLE_DIR = PACKAGE_DIR / 'g1_control' / 'example'
SAFETY_DIR = PACKAGE_DIR.parent / 'g1_safety_layer'
SAFETY_CONFIG_DIR = SAFETY_DIR / 'config'
# the sibling g1_safety_layer source, when the workspace is not sourced
if SAFETY_DIR.is_dir() and str(SAFETY_DIR) not in sys.path:
    sys.path.append(str(SAFETY_DIR))
CONTROLLER_CONFIGS = ['g1.yaml', 'g1_split.yaml', 'sim.yaml', 'sim_split.yaml']

needs_urdf = pytest.mark.skipif(
    not Path(URDF_PATH).is_file(), reason='leprechaun_g1 submodule not checked out'
)


def test_joint_groups_follow_the_g1_slot_layout():
    assert jd.NUM_MOTOR == 29
    assert jd.BODY_JOINTS == [j.name for j in BODY]
    assert jd.LOWER_BODY_INDEX == list(range(0, 12))
    assert jd.WAIST_INDEX == [12, 13, 14]
    assert jd.LEFT_ARM_INDEX == list(range(15, 22))
    assert jd.RIGHT_ARM_INDEX == list(range(22, 29))
    assert jd.UPPER_BODY_INDEX == list(range(12, 29))
    assert jd.ENABLED_JOINTS == [jd.BODY_JOINTS[i] for i in range(15, 29)]
    assert len(jd.HAND_JOINTS) == 14


@needs_urdf
@pytest.mark.parametrize('urdf', [URDF_PATH, URDF_BODY_PATH], ids=['with_hand', 'body'])
def test_models_keep_the_29_body_joints_in_slot_order(urdf):
    model = RobotModel(urdf)
    assert model.model_body.nq == 29
    assert model.model.nq == 7 + 29
    assert [n for n in model.model_body.names[1:]] == jd.BODY_JOINTS


@needs_urdf
def test_end_effector_and_sensor_frames_exist():
    model = RobotModel(URDF_PATH)
    model.init_reduced_model(jd.ENABLED_JOINTS)
    assert model.model_body_reduced.nq == 14
    for frame in (jd.LEFT_EE_FRAME, jd.RIGHT_EE_FRAME, 'imu_in_pelvis', 'imu_in_torso',
                  'd435_link', 'mid360_link', 'torso_link'):
        assert model.model_body.existFrame(frame), frame
        assert model.model_body_reduced.existFrame(frame), frame


def _pelvis_quat_wxyz(model, q, imu_wxyz):
    full = model.full_q(q, imu_quat=imu_wxyz)
    x, y, z, w = full[jd.FREEFLYER_QUAT]
    return Quaternion(w, x, y, z)


@needs_urdf
def test_pelvis_imu_passes_its_orientation_straight_through():
    model = RobotModel(URDF_PATH, imu_frame='imu_in_pelvis')
    imu = Quaternion(axis=[0.3, 1.0, 0.2], angle=0.4)
    q = np.zeros(29)
    q[jd.WAIST_INDEX[0]] = 0.5  # turning the waist must not move a pelvis IMU's pelvis
    pelvis = _pelvis_quat_wxyz(model, q, imu.elements)
    assert Quaternion.absolute_distance(pelvis, imu) < 1e-6


@needs_urdf
def test_torso_imu_is_undone_through_the_waist():
    model = RobotModel(URDF_PATH, imu_frame='imu_in_torso')
    q = np.zeros(29)
    q[jd.WAIST_INDEX[0]] = 0.5  # waist yaw
    # pelvis level and facing +x: the torso IMU then reads the waist yaw
    imu = Quaternion(axis=[0.0, 0.0, 1.0], angle=0.5)
    pelvis = _pelvis_quat_wxyz(model, q, imu.elements)
    assert Quaternion.absolute_distance(pelvis, Quaternion()) < 1e-6


@needs_urdf
def test_named_postures_are_in_limits_and_where_they_claim():
    model = RobotModel(URDF_PATH)
    model.init_reduced_model(jd.ENABLED_JOINTS)
    for name, q in NAMED_CONFIGS.items():
        assert q.shape == (14,), name
        assert model.check_within_limits_reduced(q), name
    assert set(STARTUP_SEQUENCE) <= set(NAMED_CONFIGS)

    def palms(name):
        rm, rd = model.model_body_reduced, model.data_body_reduced
        pin.framesForwardKinematics(rm, rd, NAMED_CONFIGS[name])
        return [rd.oMf[rm.getFrameId(f)].translation.copy()
                for f in (jd.LEFT_EE_FRAME, jd.RIGHT_EE_FRAME)]

    left, right = palms('home')  # forearms forward at waist height
    assert left[0] > 0.15 and right[0] > 0.15
    left, _ = palms('arms_down')  # straight down, under the shoulder
    assert abs(left[0]) < 0.05 and left[2] < 0.0
    left, right = palms('t_pose')  # straight out to the sides at shoulder height
    assert left[1] > 0.45 and right[1] < -0.45 and left[2] > 0.25


@pytest.mark.parametrize('name, domain', [('g1.yaml', 0), ('sim.yaml', 1)])
def test_g1_configs_load_and_target_the_safety_layer(name, domain):
    config = load_controller_config(name, config_dir=CONFIG_DIR)
    assert config['topics']['low_cmd'] == 'rt/safety/lowcmd_upper_in'
    assert config['network']['domain_id'] == domain
    assert config['controller']['waist_target'] == [0.0, 0.0, 0.0]
    assert config['controller']['imu_frame'] == 'imu_in_pelvis'
    assert np.all(config['gains']['kp'][jd.UPPER_BODY_INDEX] == 60.0)
    assert np.all(config['gains']['kp'][jd.LOWER_BODY_INDEX] == 0.0)
    assert np.all(config['gains']['ki'] == 0.0)
    assert config['planner']['enabled'] is False


def test_configs_cannot_bypass_the_safety_layer(tmp_path):
    base = (CONFIG_DIR / 'g1.yaml').read_text()
    direct = tmp_path / 'direct.yaml'
    direct.write_text(base.replace('low_cmd: rt/safety/lowcmd_upper_in', 'low_cmd: rt/arm_sdk'))
    with pytest.raises(ValueError, match='g1_safety_layer input'):
        load_controller_config('direct.yaml', config_dir=tmp_path)
    old = tmp_path / 'old.yaml'
    old.write_text('mode: sport\n' + base)
    with pytest.raises(ValueError, match='mode'):
        load_controller_config('old.yaml', config_dir=tmp_path)


def test_dds_interface_comes_from_g1_dds_interface(monkeypatch):
    from g1_control.utility.controller_config import _channel_factory_settings
    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.setenv('G1_DDS_INTERFACE', 'enp130s0')
    assert _channel_factory_settings(config) == (0, 'enp130s0')
    monkeypatch.delenv('G1_DDS_INTERFACE')
    assert _channel_factory_settings(config) == (0, None)


@pytest.mark.parametrize('name', CONTROLLER_CONFIGS)
def test_start_up_settings_load(name):
    config = load_controller_config(name, config_dir=CONFIG_DIR)
    controller = config['controller']
    assert controller['state_timeout'] == 5.0
    assert controller['startup_timeout'] == 30.0
    assert controller['startup_move_timeout'] == 30.0
    assert controller['follow_tolerance'] == 0.05
    # the follow gate reads the output of the safety config this one runs with
    expected = 'rt/lowcmd' if name.endswith('_split.yaml') else 'rt/arm_sdk'
    assert config['topics']['safety_out'] == expected


@pytest.mark.skipif(not SAFETY_CONFIG_DIR.is_dir(), reason='g1_safety_layer source not beside')
@pytest.mark.parametrize('safety', UPPER_CHANNEL_SAFETY_CONFIGS)
def test_paired_configs_agree_on_the_safety_output(safety):
    # by content, not by name: the topic the safety layer writes is the one the
    # paired controller config's follow gate reads
    controller = SAFETY_CONTROLLER_CONFIGS[safety]
    with (SAFETY_CONFIG_DIR / f'{safety}.yaml').open() as handle:
        low_cmd_out = yaml.safe_load(handle)['topics']['low_cmd_out']
    config = load_controller_config(f'{controller}.yaml', config_dir=CONFIG_DIR)
    assert config['topics']['safety_out'] == low_cmd_out


@pytest.mark.parametrize('topic', ['rt/safety/lowcmd_upper_in', 'rt/lowstate', 'rt/dex3/left/cmd'])
def test_safety_out_must_be_a_safety_layer_output(tmp_path, topic):
    text = (CONFIG_DIR / 'g1.yaml').read_text().replace(
        'safety_out: rt/arm_sdk', f'safety_out: {topic}')
    (tmp_path / 'bad.yaml').write_text(text)
    with pytest.raises(ValueError, match='topics.safety_out must be'):
        load_controller_config('bad.yaml', config_dir=tmp_path)


@pytest.mark.parametrize('old, new, field', [
    ('state_timeout: 5.0', 'state_timeout: 0.0', 'state_timeout'),
    ('startup_timeout: 30.0', 'startup_timeout: -1.0', 'startup_timeout'),
    ('startup_move_timeout: 30.0', 'startup_move_timeout: .inf', 'startup_move_timeout'),
    ('follow_tolerance: 0.05', 'follow_tolerance: -0.05', 'follow_tolerance'),
    ('follow_tolerance: 0.05', 'follow_tolerance: .nan', 'follow_tolerance'),
])
def test_start_up_settings_are_checked(tmp_path, old, new, field):
    (tmp_path / 'bad.yaml').write_text((CONFIG_DIR / 'g1.yaml').read_text().replace(old, new))
    with pytest.raises(ValueError, match=f'controller.{field}'):
        load_controller_config('bad.yaml', config_dir=tmp_path)


@pytest.mark.parametrize('name, speeds', [
    # deliberately conservative on the real robot: joint, palm linear, palm angular
    ('g1.yaml', (0.3, 0.15, 0.5)),
    ('g1_split.yaml', (0.3, 0.15, 0.5)),
    ('sim.yaml', (1.0, 0.5, 1.0)),
    ('sim_split.yaml', (1.0, 0.5, 1.0)),
])
def test_real_robot_speeds_are_slow_for_the_first_sessions(name, speeds):
    controller = load_controller_config(name, config_dir=CONFIG_DIR)['controller']
    assert (controller['dq_lim'], controller['v_lim'], controller['w_lim']) == speeds


@needs_urdf
@pytest.mark.parametrize('name', CONTROLLER_CONFIGS)
def test_waist_gains_are_damped_from_the_urdf_inertia(name):
    gains = load_controller_config(name, config_dir=CONFIG_DIR)['gains']
    model = RobotModel(URDF_PATH)
    # joint-space inertia at home (arms and waist at 0)
    inertia = np.diag(pin.crba(model.model_body, model.data_body, np.zeros(jd.NUM_MOTOR)))
    for i in jd.WAIST_INDEX:
        zeta = gains['kd'][i] / (2.0 * np.sqrt(gains['kp'][i] * inertia[i]))
        assert 0.4 <= zeta <= 0.5, (name, jd.BODY_JOINTS[i], zeta)
    assert np.all(gains['kd'][jd.LEFT_ARM_INDEX + jd.RIGHT_ARM_INDEX] == 1.5)

    # no safety config this one may run with clips the gains
    load_safety_config = pytest.importorskip('g1_safety_layer.core.config').load_config
    paired = [s for s in UPPER_CHANNEL_SAFETY_CONFIGS if pairing_refusal(name, s) is None]
    assert len(paired) == 1
    for safety in paired:
        limits = load_safety_config(SAFETY_CONFIG_DIR / f'{safety}.yaml')['limits']
        assert np.all(gains['kp'] <= limits['kp_clip_max']), safety
        assert np.all(gains['kd'] <= limits['kd_clip_max']), safety


def _creep(gate, measured, target, steps, step=0.01):
    '''Step an IK command toward target from the gated command, as the controller does'''
    q_cmd = np.asarray(measured(0, None), dtype=float).copy()
    gaps = []
    for k in range(steps):
        m = np.asarray(measured(k, q_cmd), dtype=float)
        q_cmd = gate.limit(q_cmd + np.clip(target - q_cmd, -step, step), m)
        gaps.append(np.max(np.abs(q_cmd[:2] - m[:2])))
    return q_cmd, gaps


def test_follow_gate_keeps_commands_near_a_robot_that_does_not_follow():
    # slot 0 commanded 0.8 rad away, slot 1 at its target; slot 2 is not gated
    target = np.array([1.0, 0.1, 1.0])
    gate = FollowGate([0, 1], 0.05)
    # at rest: the command waits 0.05 rad ahead of the robot
    q_cmd, gaps = _creep(gate, lambda k, q: [0.2, 0.1, 0.0], target, 200)
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following
    assert np.allclose(q_cmd, [0.25, 0.1, 1.0])

    # swaying 0.1 rad: the command is dragged along
    def sway(k, q):
        return [0.2 + 0.1 * np.sin(k / 10.0), 0.1, 0.0]
    _, gaps = _creep(gate, sway, target, 300)
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following

    # pushed 0.15 rad away from the target, let go, falling back: still shut
    path = np.concatenate([np.linspace(0.0, -0.15, 10), np.linspace(-0.15, 0.0, 5), np.zeros(100)])
    gate = FollowGate([0, 1], 0.05)
    _, gaps = _creep(gate, lambda k, q: [path[k], 0.1, 0.0], target, len(path))
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following


def test_follow_gate_opens_only_when_told():
    target = np.array([1.0, -1.0, 0.0])
    gate = FollowGate([0, 1], 0.05)
    lagged = {'q': np.zeros(3)}

    def follows(k, q_cmd):
        # the robot reaches each command one step later
        measured = lagged['q'].copy()
        if q_cmd is not None:
            lagged['q'] = q_cmd.copy()
        return measured
    # a robot keeping up over 1 rad of travel does not open it: the motion is no
    # evidence that the safety layer relays the commands
    q_cmd, gaps = _creep(gate, follows, target, 150)
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following
    assert np.allclose(q_cmd[:2], target[:2])

    # once open (the safety layer relays the commands), a held command never
    # follows the measured pose: a pushed or tipping joint is resisted
    gate.open()
    assert gate.following
    for measured in ([1.3, -1.2, 0.0], [0.0, 0.0, 0.0], [np.nan, 5.0, 0.0]):
        assert np.array_equal(gate.limit(target, measured), target)


def test_relays_commands_needs_the_commands_at_full_weight():
    ids = [12, 13, 14, 15]
    q_cmd = np.zeros(29)
    q_cmd[ids] = [0.0, 0.1, -0.1, 0.8]
    sample = q_cmd.copy()
    sample[ids] += [0.01, -0.02, 0.0, 0.04]
    sample[:12] = 0.7  # the legs (the walking node's in split mode) are not compared
    kp = np.zeros(29)
    kp[ids] = [200.0, 200.0, 200.0, 60.0]
    # rt/lowcmd has no weight: the robot executes it as it is
    assert relays_commands(sample, kp, None, q_cmd, ids, 0.05)
    # rt/arm_sdk: only once the weight has ramped in
    assert relays_commands(sample, kp, 1.0, q_cmd, ids, 0.05)
    for weight in (0.0, 0.5, 0.99, float('nan')):
        assert not relays_commands(sample, kp, weight, q_cmd, ids, 0.05)
    # one joint off: someone else's command, or not ours yet
    off = sample.copy()
    off[15] = 0.7
    assert not relays_commands(off, kp, 1.0, q_cmd, ids, 0.05)
    off[15] = np.nan
    assert not relays_commands(off, kp, None, q_cmd, ids, 0.05)
    # kp 0 on a slot: the safety layer's estop command (q 0, kp 0) or a default
    # LowCmd, even where our commands are near 0
    q_zero = np.zeros(29)
    for zero_kp in (np.zeros(29), np.where(np.arange(29) == 15, 0.0, kp)):
        assert not relays_commands(np.zeros(29), zero_kp, None, q_zero, ids, 0.05)
    assert relays_commands(np.zeros(29), kp, None, q_zero, ids, 0.05)


def test_follow_gate_tolerance_zero_is_off():
    gate = FollowGate([0, 1], 0.0)
    assert gate.following
    assert np.array_equal(gate.limit([1.0, 2.0], [0.0, 0.0]), [1.0, 2.0])
    with pytest.raises(ValueError):
        FollowGate([0], -0.1)


@pytest.mark.parametrize('controller, safety, kwargs, message', [
    ('g1', 'full', {}, 'safety config "full" must relay the upper-body channel'),
    ('g1', 'sim_arm_sdk', {}, 'controller config "g1" and safety config "sim_arm_sdk" mix sim'),
    ('sim_split.yaml', 'split', {}, 'mix sim and real'),
    ('sim_fast', 'arm_sdk', {}, 'mix sim and real'),
    # the real robot never runs a sim config
    ('g1', 'sim_arm_sdk', {'sim': False}, 'safety config "sim_arm_sdk" is a sim config'),
    ('sim', 'arm_sdk', {'sim': False}, 'controller config "sim" is a sim config'),
    ('sim_fast', 'split', {'sim': False}, 'controller config "sim_fast" is a sim config'),
    # nor the sim a real one
    ('g1', 'sim_arm_sdk', {'sim': True}, 'controller config "g1" is for the real robot'),
    ('sim', 'arm_sdk', {'sim': True}, 'safety config "arm_sdk" is for the real robot'),
    # walking needs split mode and the stiff split waist
    ('g1_split', 'arm_sdk', {'sim': False, 'locomotion': True},
     'needs safety_config:=split, not arm_sdk'),
    ('g1', 'split', {'sim': False, 'locomotion': True},
     'needs controller_config:=g1_split, whose stiff waist the walking policy expects, not g1'),
    ('sim', 'sim_split', {'sim': True, 'locomotion': True}, 'needs controller_config:=sim_split'),
    # each safety config runs with one controller config, walking or not
    ('g1', 'split', {}, 'safety config "split" runs with controller config "g1_split", not "g1"'),
    ('g1', 'split', {'sim': False}, 'runs with controller config "g1_split", not "g1"'),
    ('g1_split', 'arm_sdk', {'sim': False}, 'runs with controller config "g1", not "g1_split"'),
    ('sim', 'sim_split', {'sim': True}, 'runs with controller config "sim_split", not "sim"'),
    ('sim_split', 'sim_arm_sdk', {}, 'runs with controller config "sim", not "sim_split"'),
])
def test_config_pairing_refusals(controller, safety, kwargs, message):
    assert message in pairing_refusal(controller, safety, **kwargs)


def test_default_controller_config_follows_the_safety_config():
    assert [default_controller_config(s) for s in ('arm_sdk', 'split.yaml', 'sim_arm_sdk',
                                                   'config/sim_split.yaml')] == [
        'g1', 'g1_split', 'sim', 'sim_split']
    assert default_controller_config('full') is None
    for safety, controller in SAFETY_CONTROLLER_CONFIGS.items():
        assert pairing_refusal(controller, safety) is None


@pytest.mark.parametrize('controller, safety, kwargs', [
    ('g1.yaml', 'arm_sdk', {}),
    ('config/g1_split.yaml', 'split.yaml', {}),
    ('sim', 'sim_arm_sdk', {}),
    ('custom', 'sim_split', {}),
    ('g1_split', 'split', {'sim': False}),
    ('custom', 'arm_sdk', {'sim': False}),
    ('custom', 'split', {'sim': False}),
    ('custom', 'sim_arm_sdk', {'sim': True}),
    ('g1_split', 'split', {'sim': False, 'locomotion': True}),
    ('sim_split', 'sim_split', {'sim': True, 'locomotion': True}),
])
def test_config_pairings_that_may_run(controller, safety, kwargs):
    assert pairing_refusal(controller, safety, **kwargs) is None


def test_robot_safety_launch_checks_the_pairing():
    launch = pytest.importorskip('launch')
    spec = importlib.util.spec_from_file_location(
        'robot_safety_launch', PACKAGE_DIR / 'launch' / 'robot_safety_launch.py')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, KeyError) as err:  # the package is not built and sourced
        pytest.skip(f'robot_safety_launch.py needs the installed package: {err}')
    context = launch.LaunchContext()
    context.launch_configurations.update(config='g1.yaml', safety_config='sim_arm_sdk')
    with pytest.raises(ValueError, match='mix sim and real'):
        module._validate_config_compatibility(context)
    context.launch_configurations.update(config='g1_split.yaml', safety_config='split')
    assert module._validate_config_compatibility(context) == []


def test_command_tools_refuse_the_real_robot_without_the_flag(monkeypatch):
    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    initialized = []
    monkeypatch.setattr(controller_config, 'init_channel_factory_guard', initialized.append)
    # ROS_DOMAIN_ID=0, or unset with g1.yaml's domain 0
    for domain in ('0', None):
        if domain is None:
            monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)
        else:
            monkeypatch.setenv('ROS_DOMAIN_ID', domain)
        with pytest.raises(SystemExit, match='Refusing to start on DDS domain 0, the real '
                                             "robot's command bus: this tool publishes kp 0 "
                                             'everywhere.*--real-robot'):
            controller_config.init_channel_factory_for_commands(config, False, 'kp 0 everywhere')
    assert initialized == []
    # --real-robot goes on to the y/N confirmation; other domains need no flag
    controller_config.init_channel_factory_for_commands(config, True, 'kp 0 everywhere')
    monkeypatch.setenv('ROS_DOMAIN_ID', '1')
    controller_config.init_channel_factory_for_commands(config, False, 'kp 0 everywhere')
    assert len(initialized) == 2


def _load_example(name):
    spec = importlib.util.spec_from_file_location(f'example_{name[:-3]}', EXAMPLE_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name, start', [
    ('motor_viewer.py', lambda m: m.main(0.5, 30.0)),
    ('motor_viewer_tui.py', lambda m: m.main(0.5, 30.0, 0.8, 0.9)),
    ('arm_controller_goto.py', lambda m: m.main()),
    ('arm_controller_gui.py', lambda m: m.main()),
    ('frame_controller_goto.py', lambda m: m.main()),
    ('frame_task_trajectory.py', lambda m: m.init_frame_controller('g1.yaml')),
])
def test_example_tools_that_publish_refuse_domain_0(monkeypatch, name, start):
    module = _load_example(name)
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.setattr(controller_config, 'init_channel_factory_guard',
                        lambda config: pytest.fail('DDS initialized on domain 0'))
    with pytest.raises(SystemExit, match='--real-robot'):
        start(module)


@pytest.mark.parametrize('name, start', [
    ('motor_viewer.py', lambda m: m.main(None, 30.0)),
    ('motor_viewer_tui.py', lambda m: m.main(None, 30.0, 0.8, 0.9)),
])
def test_motor_viewers_only_listen_without_damping(monkeypatch, name, start):
    module = _load_example(name)
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')

    class Listening(Exception):
        pass

    def guard(config):
        raise Listening
    # listener only: on to the y/N confirmation, no refusal
    monkeypatch.setattr(module, 'init_channel_factory_guard', guard)
    with pytest.raises(Listening):
        start(module)
