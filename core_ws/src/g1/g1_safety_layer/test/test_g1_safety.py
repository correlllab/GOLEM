import os
import sys
import time
import signal
import threading
from pathlib import Path

import yaml
import numpy as np
import pytest

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault

from g1_description.joints import BODY
from g1_safety_layer.core import console
from g1_safety_layer.core import config as config_module
from g1_safety_layer.core.clock import watchdog_time
from g1_safety_layer.core.config import LOG_DIR_ENV, load_config
from g1_safety_layer.core.safety_checks import (
    CmdValidationError,
    EStopTriggered,
    check_estop_limits,
    check_waist_kp,
    clipped_gains,
    engage_refusal,
    make_estop_cmd,
)
from g1_safety_layer.core.joint_limits import ARM_SDK_SLOTS, ARM_SDK_WEIGHT_SLOT, MOTOR_COUNT
from g1_safety_layer.core.arm_sdk import engage_weight, release_weight, with_weight, zero_legs

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
CONFIGS = sorted(CONFIG_DIR.glob('*.yaml'))


def test_all_configs_are_present():
    names = {p.name for p in CONFIGS}
    assert names == {'arm_sdk.yaml', 'full.yaml', 'split.yaml', 'sim_arm_sdk.yaml', 'sim_split.yaml'}


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.name)
def test_config_limits_cover_the_29_g1_motors(path):
    config = load_config(path)
    limits = config['limits']
    for key in ('q_clip_limits', 'q_estop_limits'):
        assert limits[key].shape == (MOTOR_COUNT, 2)
        assert np.all(limits[key][:, 0] < limits[key][:, 1])
    for key in ('dq_clip_limits', 'tau_clip_limits', 'dq_estop_limits', 'tau_estop_limits'):
        assert limits[key].shape == (MOTOR_COUNT,)
    expected_out = 'rt/arm_sdk' if config['mode'] == 'arm_sdk_mode' else 'rt/lowcmd'
    assert config['topics']['low_cmd_out'] == expected_out


def test_arm_sdk_estop_limits_are_the_urdf_limits():
    limits = load_config(CONFIG_DIR / 'arm_sdk.yaml')['limits']
    assert limits['q_estop_limits'].tolist() == [[j.lower, j.upper] for j in BODY]
    # first-session torques on waist + arms (N*m): waist yaw, roll, pitch, then
    # each arm; the legs are not checked in arm_sdk_mode
    arm = [12.5, 12.5, 12.5, 12.5, 12.5, 2.5, 2.5]
    assert np.allclose(limits['tau_estop_limits'][12:29], [44.0, 24.5, 24.5] + arm + arm)


def test_old_external_estop_section_is_rejected(tmp_path):
    cfg = tmp_path / 'old.yaml'
    cfg.write_text('mode: split_mode\nestop:\n  enabled: true\n')
    with pytest.raises(ValueError, match='external estop'):
        load_config(cfg)


def test_lower_stale_estop_seconds_per_config(tmp_path):
    # the sim can run slower than real time, so its watchdog is looser
    expected = {'split.yaml': 0.2, 'full.yaml': 0.2, 'sim_split.yaml': 1.0}
    for name, seconds in expected.items():
        control = load_config(CONFIG_DIR / name)['control']
        assert control['lower_stale_estop_seconds'] == seconds, name
    cfg = tmp_path / 'default.yaml'
    cfg.write_text('robot: real\nmode: full_body_mode\n')
    assert load_config(cfg)['control']['lower_stale_estop_seconds'] == 0.2


@pytest.mark.parametrize('value', ['0', '-0.1', '.nan', '.inf'])
def test_lower_stale_estop_seconds_must_be_positive_and_finite(tmp_path, value):
    cfg = tmp_path / 'bad.yaml'
    cfg.write_text(f'mode: split_mode\ncontrol:\n  lower_stale_estop_seconds: {value}\n')
    with pytest.raises(ValueError, match='lower_stale_estop_seconds'):
        load_config(cfg)


def test_estop_response_defaults_to_zero(tmp_path):
    # today's behavior: every motor off; the shipped kd is only used by damp
    for path in CONFIGS:
        response = load_config(path)['estop_response']
        assert response['mode'] == 'zero', path.name
    for name in ('split.yaml', 'full.yaml', 'sim_split.yaml'):
        assert load_config(CONFIG_DIR / name)['estop_response']['kd'].shape == (MOTOR_COUNT,)
    cfg = tmp_path / 'default.yaml'
    cfg.write_text('robot: real\nmode: split_mode\n')
    assert load_config(cfg)['estop_response'] == {'mode': 'zero', 'kd': None}


def test_estop_response_damp_takes_a_scalar_or_per_joint_kd(tmp_path):
    cfg = tmp_path / 'damp.yaml'
    cfg.write_text('robot: real\nmode: full_body_mode\nestop_response:\n  mode: damp\n  kd: 3\n')
    response = load_config(cfg)['estop_response']
    assert response['mode'] == 'damp'
    assert response['kd'].tolist() == [3.0] * MOTOR_COUNT
    kd = [0.1 * i for i in range(MOTOR_COUNT)]
    cfg.write_text(
        f'robot: real\nmode: full_body_mode\nestop_response:\n  mode: damp\n  kd: {kd}\n'
    )
    assert load_config(cfg)['estop_response']['kd'].tolist() == kd


@pytest.mark.parametrize('section, match', [
    ('mode: brake', 'mode must be zero or damp'),
    ('mode: damp', 'kd is required'),
    ('mode: damp\n  kd: 10.5', 'must not exceed limits.clip.kd_max'),
    ('mode: damp\n  kd: [1.0, 2.0]', 'kd must be a number or a list'),
    ('mode: damp\n  kd: .nan', 'kd must be nonnegative and finite'),
    ('mode: zero\n  kd: -1.0', 'kd must be nonnegative and finite'),  # checked when set
])
def test_estop_response_is_validated(tmp_path, section, match):
    cfg = tmp_path / 'bad.yaml'
    cfg.write_text(
        'mode: split_mode\nlimits:\n  clip:\n    kd_max: 10.0\n'
        f'estop_response:\n  {section}\n'
    )
    with pytest.raises(ValueError, match=match):
        load_config(cfg)


def test_estop_cmd_zeroes_or_damps_every_motor():
    template = LowCmdDefault()
    template.mode_machine = 5
    template.motor_cmd[3].mode = 1
    template.motor_cmd[3].q = 0.5
    template.motor_cmd[3].kp = 60.0
    kd = np.arange(MOTOR_COUNT) * 0.1
    zero = make_estop_cmd(template)
    damp = make_estop_cmd(template, kd)
    for i in range(MOTOR_COUNT):
        z, d = zero.motor_cmd[i], damp.motor_cmd[i]
        assert (z.mode, z.q, z.dq, z.tau, z.kp, z.kd) == (0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert (d.mode, d.q, d.dq, d.tau, d.kp) == (1, 0.0, 0.0, 0.0, 0.0)
        assert d.kd == pytest.approx(kd[i])
    assert zero.mode_machine == damp.mode_machine == 5  # headers kept
    assert template.motor_cmd[3].kp == 60.0  # template untouched


def test_clipped_gains_lists_the_gains_above_the_max():
    limits = load_config(CONFIG_DIR / 'split.yaml')['limits']  # kp_max 200, kd_max 10
    cmd = LowCmdDefault()
    cmd.motor_cmd[3].kp = 300.0  # left knee above kp_max
    cmd.motor_cmd[3].kd = 10.0  # at kd_max: not clipped
    cmd.motor_cmd[18].kd = 12.0  # left elbow above kd_max
    assert clipped_gains(cmd, limits, range(MOTOR_COUNT)) == [
        (3, 'kp', 300.0, 200.0),
        (18, 'kd', 12.0, 10.0),
    ]
    assert clipped_gains(cmd, limits, range(12)) == [(3, 'kp', 300.0, 200.0)]  # legs only


def test_engage_and_release_ramps():
    assert engage_weight(10.0, 10.0, 1.0) == 0.0
    assert engage_weight(10.5, 10.0, 1.0) == pytest.approx(0.5)
    assert engage_weight(12.0, 10.0, 1.0) == 1.0
    assert engage_weight(10.0, 10.0, 0.0) == 1.0
    # release starts from whatever weight was reached and ends at 0
    assert release_weight(20.0, 20.0, 0.6, 1.0) == pytest.approx(0.6)
    assert release_weight(20.5, 20.0, 0.6, 1.0) == pytest.approx(0.3)
    assert release_weight(25.0, 20.0, 0.6, 1.0) == 0.0


def test_with_weight_only_touches_the_weight_slot():
    cmd = LowCmdDefault()
    cmd.motor_cmd[15].q = 0.3
    cmd.motor_cmd[ARM_SDK_WEIGHT_SLOT].kp = 5.0
    out = with_weight(cmd, 0.25)
    assert out.motor_cmd[ARM_SDK_WEIGHT_SLOT].q == pytest.approx(0.25)
    assert out.motor_cmd[ARM_SDK_WEIGHT_SLOT].kp == 0.0
    assert out.motor_cmd[15].q == pytest.approx(0.3)
    assert cmd.motor_cmd[ARM_SDK_WEIGHT_SLOT].q == 0.0  # input untouched


def test_state_check_can_ignore_the_legs():
    limits = load_config(CONFIG_DIR / 'arm_sdk.yaml')['limits']
    state = LowStateDefault()
    state.motor_state[3].q = 10.0  # left knee far out of range (a leg slot)
    check_estop_limits(state, limits, ARM_SDK_SLOTS)
    with pytest.raises(EStopTriggered, match='left_knee_joint'):
        check_estop_limits(state, limits)
    state.motor_state[18].q = 10.0  # left elbow (an arm slot)
    with pytest.raises(EStopTriggered, match='left_elbow_joint'):
        check_estop_limits(state, limits, ARM_SDK_SLOTS)


def test_dds_interface_comes_from_g1_dds_interface(monkeypatch):
    from g1_safety_layer.core.config import _channel_factory_settings
    config = load_config(CONFIG_DIR / 'arm_sdk.yaml')
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    monkeypatch.setenv('G1_DDS_INTERFACE', 'enp130s0')
    assert _channel_factory_settings(config) == (0, 'enp130s0')
    monkeypatch.delenv('G1_DDS_INTERFACE')
    assert _channel_factory_settings(config) == (0, None)


def test_every_config_declares_real_or_sim():
    for path in CONFIGS:
        expected = 'sim' if path.name.startswith('sim_') else 'real'
        assert yaml.safe_load(path.read_text())['robot'] == expected, path.name
        assert load_config(path)['robot'] == expected, path.name


@pytest.mark.parametrize('text, match', [
    ('mode: split_mode\n', 'must declare robot: real or robot: sim'),
    ('robot: gantry\nmode: split_mode\n', "robot must be real or sim, not 'gantry'"),
    ('robot: sim\nmode: split_mode\nnetwork:\n  domain_id: 0\n',
     'must not set network.domain_id 0'),
    ('robot: sim\nmode: split_mode\n', 'must not set network.domain_id 0'),  # the default is 0
])
def test_robot_declaration_is_validated(tmp_path, text, match):
    cfg = tmp_path / 'bad.yaml'
    cfg.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_config(cfg)


def test_sim_configs_refuse_dds_domain_0(monkeypatch):
    # a stub, so that nothing here can start DDS, whatever the guard does
    started = []
    monkeypatch.setattr(config_module, 'ChannelFactoryInitialize', lambda *a: started.append(a))
    monkeypatch.delenv('G1_DDS_INTERFACE', raising=False)
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    for name in ('sim_arm_sdk.yaml', 'sim_split.yaml'):
        config = load_config(CONFIG_DIR / name)
        for init in (config_module.init_channel_factory, config_module.init_channel_factory_guard):
            with pytest.raises(ValueError, match='refuses DDS domain 0'):
                init(config)
    assert started == []
    # real configs still reach domain 0, and sim configs their own domain
    real = load_config(CONFIG_DIR / 'arm_sdk.yaml')
    assert config_module._channel_factory_settings(real) == (0, None)
    monkeypatch.delenv('ROS_DOMAIN_ID')
    config_module.init_channel_factory(load_config(CONFIG_DIR / 'sim_arm_sdk.yaml'))
    assert started == [(1,)]


def test_engage_gate_and_state_watchdog_settings(tmp_path):
    # the sim can run slower than real time: its rt/lowstate ages are relaxed
    ages = {'arm_sdk.yaml': 0.1, 'split.yaml': 0.1, 'full.yaml': 0.1,
            'sim_arm_sdk.yaml': 1.0, 'sim_split.yaml': 1.0}
    for name, age in ages.items():
        config = load_config(CONFIG_DIR / name)
        assert config['engage'] == {
            'max_state_age_seconds': age, 'mode_machine': 5, 'max_position_error': 0.15
        }, name
        assert config['control']['state_stale_estop_seconds'] == age, name
    cfg = tmp_path / 'default.yaml'
    cfg.write_text('robot: real\nmode: arm_sdk_mode\n')
    config = load_config(cfg)
    assert config['engage'] == {
        'max_state_age_seconds': 0.1, 'mode_machine': 5, 'max_position_error': 0.15
    }
    assert config['control']['state_stale_estop_seconds'] == 0.1
    assert config['arm_sdk']['waist_kp_min'] == 40.0


@pytest.mark.parametrize('section, match', [
    ('control:\n  state_stale_estop_seconds: 0', 'state_stale_estop_seconds must be positive'),
    ('engage:\n  max_state_age_seconds: .nan', 'max_state_age_seconds must be positive'),
    ('engage:\n  max_position_error: -0.1', 'max_position_error must be positive'),
    ('engage:\n  mode_machine: 5.0', 'mode_machine must be an integer'),
    ('engage:\n  mode_machine: true', 'mode_machine must be an integer'),
    ('arm_sdk:\n  waist_kp_min: -1', 'waist_kp_min must be nonnegative'),
    ('arm_sdk:\n  waist_kp_min: 250', 'must not exceed the waist limits.clip.kp_max'),
])
def test_engage_gate_and_waist_settings_are_validated(tmp_path, section, match):
    cfg = tmp_path / 'bad.yaml'
    cfg.write_text(
        f'robot: real\nmode: arm_sdk_mode\nlimits:\n  clip:\n    kp_max: 200\n{section}\n'
    )
    with pytest.raises(ValueError, match=match):
        load_config(cfg)


def test_engage_refusal_needs_fresh_state_its_mode_machine_and_a_close_command():
    gate = {'max_state_age_seconds': 0.1, 'mode_machine': 5, 'max_position_error': 0.15}
    cmd = LowCmdDefault()
    cmd.motor_cmd[18].q = 0.5  # left elbow
    q = np.zeros(MOTOR_COUNT)
    q[18] = 0.4
    assert engage_refusal(cmd, ARM_SDK_SLOTS, q, 0.05, 5, gate) is None
    assert engage_refusal(cmd, ARM_SDK_SLOTS, None, None, 5, gate) == 'no rt/lowstate received yet'
    assert engage_refusal(cmd, ARM_SDK_SLOTS, q, 0.25, 5, gate) == (
        'rt/lowstate is 0.250 s old (> 0.1 s)'
    )
    assert engage_refusal(cmd, ARM_SDK_SLOTS, q, 0.05, 6, gate) == (
        'rt/lowstate mode_machine is 6, expected 5'
    )
    q[18] = 0.3
    assert engage_refusal(cmd, ARM_SDK_SLOTS, q, 0.05, 5, gate) == (
        'motor 18 (left_elbow_joint) command q 0.500 is 0.200 rad from rt/lowstate '
        'q 0.300 (> 0.15 rad)'
    )
    # only the given slots count: arm_sdk_mode ignores a leg far from its command
    q[18] = 0.5
    q[3] = 1.0
    assert engage_refusal(cmd, ARM_SDK_SLOTS, q, 0.05, 5, gate) is None
    assert 'motor 3 (left_knee_joint)' in engage_refusal(cmd, range(MOTOR_COUNT), q, 0.05, 5, gate)
    q[20] = float('nan')  # a NaN must not pass as "not too far"
    assert engage_refusal(cmd, ARM_SDK_SLOTS, q, 0.05, 5, gate) == (
        'motor 20 (left_wrist_pitch_joint) has a non-finite q'
    )


def test_waist_kp_below_the_floor_is_rejected():
    cmd = LowCmdDefault()
    for i in (12, 13, 14):
        cmd.motor_cmd[i].kp = 60.0
    check_waist_kp(cmd, 40.0)
    cmd.motor_cmd[14].kp = 39.0
    match = r'motor 14 \(waist_pitch_joint\) kp 39.0 is below'
    with pytest.raises(CmdValidationError, match=match):
        check_waist_kp(cmd, 40.0)


def test_zero_legs_clears_slots_0_to_11_only():
    cmd = LowCmdDefault()
    for i in range(MOTOR_COUNT):
        m = cmd.motor_cmd[i]
        m.mode, m.q, m.dq, m.tau, m.kp, m.kd = 1, 0.3, 0.1, 2.0, 50.0, 1.0
    assert zero_legs(cmd) is cmd
    for i in range(MOTOR_COUNT):
        m = cmd.motor_cmd[i]
        fields = (m.mode, m.q, m.dq, m.tau, m.kp, m.kd)
        if i < 12:
            assert fields == (0, 0.0, 0.0, 0.0, 0.0, 0.0), i
        else:
            assert fields == pytest.approx((1, 0.3, 0.1, 2.0, 50.0, 1.0)), i


def test_watchdog_clock_is_boottime():
    if not hasattr(time, 'CLOCK_BOOTTIME'):
        pytest.skip('CLOCK_BOOTTIME is Linux only')
    before = time.clock_gettime(time.CLOCK_BOOTTIME)
    now = watchdog_time()
    after = time.clock_gettime(time.CLOCK_BOOTTIME)
    assert before <= now <= after
    # and it is the clock the safety layer times its watchdogs and ramps with
    from g1_safety_layer.core import safety_layer
    assert safety_layer.watchdog_time is watchdog_time


def test_log_dir_comes_from_g1_safety_log_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(LOG_DIR_ENV, raising=False)
    default = load_config(CONFIG_DIR / 'arm_sdk.yaml')['logging']['base_dir']
    assert default == '/tmp/g1_safety_layer'
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    base_dir = Path(load_config(CONFIG_DIR / 'arm_sdk.yaml')['logging']['base_dir'])
    # a directory of its own per run: ChunkLogger clears its directory at start
    assert base_dir.parent == tmp_path
    assert base_dir.name.endswith(f'-{os.getpid()}')


def test_arm_sdk_first_session_defaults():
    config = load_config(CONFIG_DIR / 'arm_sdk.yaml')
    assert config['arm_sdk']['release_seconds'] == 3.0
    assert config['arm_sdk']['waist_kp_min'] == 40.0
    limits = config['limits']
    upper = list(ARM_SDK_SLOTS)
    # every waist and arm joint: commands clipped at the controller's 1 rad/s
    # cap, the estop at 2 rad/s
    assert np.allclose(limits['dq_clip_limits'][upper], 1.0, atol=0.002)
    assert np.allclose(limits['dq_estop_limits'][upper], 2.0, atol=0.002)
    # the largest static gravity load (N*m) in the controller's postures, pelvis
    # upright and no payload (pinocchio on g1_29dof_with_hand_rev_1_0.urdf), is
    # under the feedforward clip, which is under the estop
    gravity = {14: 18.16, 15: 8.50, 16: 8.59, 18: 3.41, 20: 1.22, 21: 0.76,
               22: 8.50, 23: 8.59, 25: 3.41, 27: 1.22, 28: 0.76}
    for i in upper:
        assert limits['tau_clip_limits'][i] < limits['tau_estop_limits'][i], i
        assert gravity.get(i, 0.0) < limits['tau_clip_limits'][i], i


class _BrokenStdout:
    '''A stdout whose reader is gone, without a file descriptor'''
    def __init__(self) -> None:
        self.writes = 0

    def write(self, text: str) -> int:
        self.writes += 1
        raise BrokenPipeError(32, 'Broken pipe')

    def flush(self) -> None:
        pass


class _StalledStdout:
    '''A stdout whose reader stopped reading until release is set; keeps each write'''
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.entered.set()
        self.release.wait()
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        pass


def test_console_drops_every_line_once_stdout_breaks(monkeypatch):
    broken = _BrokenStdout()
    monkeypatch.setattr(sys, 'stdout', broken)
    out = console.Console()
    for i in range(3):
        out.write_line(f'line {i}')
    assert out.drain(1.0)
    # gone for good: one failed write, the later lines dropped unwritten
    assert broken.writes == 1


def test_console_never_blocks_on_a_stalled_stdout(monkeypatch):
    stalled = _StalledStdout()
    monkeypatch.setattr(sys, 'stdout', stalled)
    monkeypatch.setattr(console, 'MAX_PENDING_LINES', 3)
    out = console.Console()

    def write_lines(first: int, last: int) -> None:
        '''Write lines first..last from a thread; fail if that blocks'''
        writer = threading.Thread(
            target=lambda: [out.write_line(f'line {i}') for i in range(first, last + 1)]
        )
        writer.start()
        writer.join(1.0)
        assert not writer.is_alive(), 'write_line blocked on a stalled stdout'

    try:
        write_lines(0, 0)
        assert stalled.entered.wait(1.0)
        # line 0 is being written: lines 1 and 2 fill the queue, 3 to 9 are dropped
        write_lines(1, 9)
        assert not out.drain(0.1)
    finally:
        stalled.release.set()
    assert out.drain(1.0)
    assert stalled.writes == [
        'line 0\n', '{"count": 7, "event": "stdout_lines_dropped"}\n', 'line 1\n', 'line 2\n'
    ]


def test_sighup_hands_control_back_too():
    from g1_safety_layer.core.exit_signals import EXIT_SIGNALS
    assert set(EXIT_SIGNALS) == {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}


class _ExitAtOnce:
    '''An ExitRequest whose SIGTERM has already arrived; installs no handler'''
    signal_name = 'SIGTERM'

    def install(self) -> None:
        pass

    def is_set(self) -> bool:
        return True


def _stub_node_main(monkeypatch, handed_back: bool = True):
    '''
    Stub rclpy, SafetyNode and the signal handlers of safety_node.main, so that
    nothing here can start DDS; return the module and the rclpy.init calls
    '''
    pytest.importorskip('rclpy')
    from g1_safety_layer.ros2 import safety_node
    inits = []

    class FakeNode:
        failed = False

        def __init__(self, config, config_path) -> None:
            self.handed_back = False

        def running(self) -> bool:
            return True

        def destroy_node(self, reason: str) -> None:
            self.handed_back = handed_back

    monkeypatch.setattr(safety_node.rclpy, 'init', lambda **kwargs: inits.append(kwargs))
    monkeypatch.setattr(safety_node.rclpy, 'spin_once', lambda *args, **kwargs: None)
    monkeypatch.setattr(safety_node.rclpy, 'ok', lambda: False)
    monkeypatch.setattr(safety_node, 'SafetyNode', FakeNode)
    monkeypatch.setattr(safety_node, 'ExitRequest', _ExitAtOnce)
    monkeypatch.delenv('G1_DDS_INTERFACE', raising=False)
    return safety_node, inits


def test_the_node_refuses_a_sim_config_on_domain_0_before_rclpy_starts(monkeypatch):
    safety_node, inits = _stub_node_main(monkeypatch)
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    for name in ('sim_arm_sdk.yaml', 'sim_split.yaml'):
        with pytest.raises(ValueError, match='refuses DDS domain 0'):
            safety_node.main(['--config', str(CONFIG_DIR / name)])
    assert inits == []
    # otherwise rclpy joins the domain the Unitree channels use, also with
    # ROS_DOMAIN_ID unset (where rclpy alone would join domain 0)
    sim = ['--config', str(CONFIG_DIR / 'sim_arm_sdk.yaml')]
    monkeypatch.setenv('ROS_DOMAIN_ID', '97')
    assert safety_node.main(sim) == 0
    monkeypatch.delenv('ROS_DOMAIN_ID')
    assert safety_node.main(sim) == 0
    assert [call['domain_id'] for call in inits] == [97, 1]


@pytest.mark.parametrize('handed_back, code', [(True, 0), (False, 1)])
def test_the_node_exits_non_zero_when_the_hand_back_does_not_finish(
        monkeypatch, handed_back, code):
    safety_node, _ = _stub_node_main(monkeypatch, handed_back=handed_back)
    monkeypatch.setenv('ROS_DOMAIN_ID', '97')
    assert safety_node.main(['--config', str(CONFIG_DIR / 'sim_arm_sdk.yaml')]) == code


@pytest.mark.parametrize('handed_back, code', [(True, 0), (False, 1)])
def test_the_script_exits_non_zero_when_the_hand_back_does_not_finish(
        monkeypatch, handed_back, code):
    from g1_safety_layer.script import safety_layer_main as script

    class FakeLayer:
        def __init__(self, config) -> None:
            self._stopped = threading.Event()

        def start(self) -> None:
            self._stopped.wait(5.0)

        def shutdown(self, reason: str) -> bool:
            self._stopped.set()
            return handed_back

    # nothing here can start DDS
    monkeypatch.setattr(script, 'init_channel_factory_guard', lambda config: None)
    monkeypatch.setattr(script, 'SafetyLayer', FakeLayer)
    monkeypatch.setattr(script, 'ExitRequest', _ExitAtOnce)
    assert script.main(str(CONFIG_DIR / 'arm_sdk.yaml')) == code
