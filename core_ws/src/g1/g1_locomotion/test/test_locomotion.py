'''
The walking node's logic with no DDS or ROS: config, slots, state, commands,
startup, the tick cadence, the latency and faults
'''

import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault

import unitree_sdk2py.core.channel
from g1_locomotion.config import check_config, init_channel_factory, load_config
from g1_locomotion.locomotion import (
    FAULTED,
    POLICY_DT,
    POLICY_SLOTS,
    RAMPING,
    STATS_SECONDS,
    STEADY_SECONDS,
    TICK_MODULO,
    WAITING,
    WALKING,
    CommandFilter,
    LocomotionController,
    SourceClock,
    State,
    gravity_z,
    load_policy,
    policy_slots,
    read_state,
    tick_delta,
)

# the package's policy/ (the containers also mount it at /home/code/g1_locomotion)
POLICY_DIR = Path(__file__).resolve().parents[1] / 'policy'
CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
SAMPLE_DT = 0.002  # rt/lowstate at 500 Hz
TICK_S = 0.001  # its tick counts ms
# rt/lowstate samples per policy step, and to start, when every one arrives
SAMPLES_PER_STEP = 10
SAMPLES_TO_START = round(STEADY_SECONDS / SAMPLE_DT)


def _raw_config(name='g1.yaml'):
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def _tilted(degrees):
    '''Pelvis quaternion (w, x, y, z) rolled by `degrees` about x'''
    half = math.radians(degrees) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)


class FakeSession:
    '''Stands in for the ONNX session: records each input, returns zero actions, runs `hook`'''

    def __init__(self, hook=None):
        self.inputs = []
        self.hook = hook

    def run(self, output_names, feed):
        (x,) = feed.values()
        self.inputs.append(x.copy())
        if self.hook is not None:
            return [np.array([self.hook()], dtype=np.float32)]
        return [np.zeros((1, 12), dtype=np.float32)]


class Clock:
    '''A fake monotonic clock'''

    def __init__(self):
        self.t = 100.0


@pytest.fixture(scope='module')
def trained():
    '''The policy's trained vx, vy and yaw ranges'''
    return load_policy(str(POLICY_DIR)).config['cmd_range']


@pytest.fixture
def policy():
    return load_policy(str(POLICY_DIR))


def _standing(policy, legs=None, quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)):
    '''The State of a still robot, legs at `legs` (default: the default pose), the rest at 0'''
    q = np.zeros(len(POLICY_SLOTS))
    q[:12] = policy.default_angles if legs is None else legs
    return State(np.array(quat, dtype=np.float64), np.array(gyro, dtype=np.float64),
                 q, np.zeros(len(POLICY_SLOTS)))


class Rig:
    '''
    A LocomotionController on a fake clock, fed samples 2 ms (2 ticks) apart,
    and published as the driver does: only if latency_ok, the latency counted
    from the sample's arrival (no delivery delay)
    '''

    def __init__(self, policy, trained, ramp_seconds=0.1, tick=1000):
        raw = _raw_config()
        raw['startup']['ramp_seconds'] = ramp_seconds
        self.config = check_config(raw, trained)
        self.policy = policy
        self.session = policy.session = FakeSession()
        self.clock = Clock()
        self.tick = tick
        self.log = []
        self.steps = []  # the ticks of the published commands
        self.controller = LocomotionController(
            policy, self.config, log=lambda level, message: self.log.append((level, message)))

    def feed(self, state, n, upper=True, dt=SAMPLE_DT, ticks=None, age=0.0):
        '''
        n samples dt s apart, their ticks `ticks` apart (dt in ms by default),
        each taken age s after it was written; returns what was published for
        each (None: silent)
        '''
        out = []
        for _ in range(n):
            self.clock.t += dt
            step = round(dt / TICK_S) if ticks is None else ticks
            self.tick = (self.tick + step) % TICK_MODULO
            if upper:
                self.controller.on_upper_cmd(self.clock.t)
            arrival = self.clock.t
            targets = self.controller.on_lowstate(state, self.tick, arrival, age)
            if targets is not None and not self.controller.latency_ok(self.clock.t - arrival):
                targets = None
            if targets is not None:
                self.steps.append(self.tick)
                self.controller.log_timing(self.clock.t)
            out.append(targets)
        return out

    def walk(self, state):
        '''Run the startup (a 0.1 s ramp) until the policy has stepped once'''
        for _ in range(2 * SAMPLES_TO_START):
            self.feed(state, 1)
            if self.session.inputs:
                return
        raise AssertionError(f'never walked: {self.controller.phase}, {self.log}')


def _messages(log, level=None):
    return [m for lvl, m in log if level is None or lvl == level]


def test_configs_load(trained):
    real = load_config(CONFIG_DIR / 'g1.yaml', trained)
    assert real['topics'] == {'low_state': 'rt/lowstate', 'low_cmd': 'rt/safety/lowcmd_lower_in',
                              'upper_cmd': 'rt/safety/lowcmd_upper_in'}
    assert real['network'] == {'domain_id': 0, 'interface': None}
    assert real['startup']['ramp_seconds'] == 3.0
    limits = [real['commands'][k] for k in ('vx', 'vy', 'yaw')]
    assert limits == [(-0.3, 0.5), (-0.3, 0.3), (-0.6, 0.6)]
    assert real['commands']['timeout'] == 0.5
    assert real['faults'] == {'max_state_gap': 0.05, 'max_latency': 0.025}
    assert real['tick_seconds'] == 0.001
    assert real['startup']['tick_rate'] == (0.9, 1.1)

    sim = load_config(CONFIG_DIR / 'sim.yaml', trained)
    assert sim['network']['domain_id'] == 1 and sim['startup']['ramp_seconds'] < 1.0
    # the sim's tick is ms of sim time, and its latency limit the robot's
    assert sim['tick_seconds'] == 0.001 and sim['faults']['max_latency'] == 0.025
    # its real-time factor may drop well below 1
    assert sim['startup']['tick_rate'] == (0.25, 1.25)
    # the sim uses the trained ranges
    assert [list(sim['commands'][k]) for k in ('vx', 'vy', 'yaw')] == trained


@pytest.mark.parametrize('section, key, value, message', [
    ('topics', 'low_cmd', 'rt/lowcmd',
     'topics.low_cmd must be a g1_safety_layer input (rt/safety/...), got rt/lowcmd'),
    ('topics', 'low_cmd', 'rt/safety/lowcmd_upper_in',
     'topics.low_cmd and topics.upper_cmd must differ'),
    (None, 'imu_frame', 'torso', "imu_frame must be one of ['pelvis'], got 'torso'"),
    ('commands', 'vx', [-0.3, 1.5],
     'commands.vx [-0.3, 1.5] must include 0 and lie inside the trained range [-0.8, 1.2]'),
    ('commands', 'yaw', [-0.9, 0.6], 'commands.yaw [-0.9, 0.6] must include 0'),
    ('commands', 'vy', [0.1, 0.3], 'commands.vy [0.1, 0.3] must include 0'),
    ('commands', 'vx', [0.5], 'commands.vx must be a list of 2 numbers'),
    ('commands', 'accel', [0.5, 0.0, 1.0], 'commands.accel (vy) must be positive'),
    ('commands', 'timeout', True, 'commands.timeout must be a number'),
    ('startup', 'ramp_seconds', 0.0, 'startup.ramp_seconds must be positive'),
    ('startup', 'tick_rate', [0.9, 0.95],
     'startup.tick_rate [0.9, 0.95] must be [low, high] with 0 < low <= 1 <= high'),
    ('startup', 'tick_rate', [0.0, 1.1], 'must be [low, high] with 0 < low <= 1 <= high'),
    ('startup', 'tick_rate', 1.0, 'startup.tick_rate must be a list of 2 numbers'),
    ('faults', 'max_state_gap', math.inf, 'faults.max_state_gap must be positive and finite'),
    ('faults', 'max_latency', 0, 'faults.max_latency must be positive'),
    (None, 'tick_seconds', 0.003, 'tick_seconds must divide the 0.02 s policy period, got 0.003'),
    (None, 'tick_seconds', 0.04, 'tick_seconds must divide the 0.02 s policy period, got 0.04'),
    (None, 'tick_seconds', '1ms', "tick_seconds must be a number, got '1ms'"),
    ('network', 'domain_id', -1, 'network.domain_id must be a nonnegative integer'),
])
def test_config_rejects_bad_values(trained, section, key, value, message):
    raw = _raw_config()
    (raw if section is None else raw[section])[key] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        check_config(raw, trained)


def test_config_rejects_missing_and_unknown_keys(trained):
    raw = _raw_config()
    del raw['faults']['max_state_gap']
    message = "faults: missing keys ['max_state_gap'], unknown keys []"
    with pytest.raises(ValueError, match=re.escape(message)):
        check_config(raw, trained)
    raw = _raw_config()
    raw['startup']['ramp_second'] = 1.0  # a typo must not fall back to a default
    with pytest.raises(ValueError, match=re.escape("unknown keys ['ramp_second']")):
        check_config(raw, trained)
    raw = _raw_config()
    raw['estop'] = {}
    message = "config: missing keys [], unknown keys ['estop']"
    with pytest.raises(ValueError, match=re.escape(message)):
        check_config(raw, trained)


def test_dds_domain_and_interface(trained, monkeypatch):
    # the SDK call is faked: nothing here may start DDS, least of all on domain 0
    calls = []
    monkeypatch.setattr(unitree_sdk2py.core.channel, 'ChannelFactoryInitialize',
                        lambda *args: calls.append(args))
    real = load_config(CONFIG_DIR / 'g1.yaml', trained)
    sim = load_config(CONFIG_DIR / 'sim.yaml', trained)
    monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)
    monkeypatch.delenv('G1_DDS_INTERFACE', raising=False)
    init_channel_factory(sim)
    monkeypatch.setenv('ROS_DOMAIN_ID', '97')
    monkeypatch.setenv('G1_DDS_INTERFACE', 'lo')
    init_channel_factory(real)
    assert calls == [(1,), (97, 'lo')]
    # the sim's config never drives the real robot
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    message = ('ROS_DOMAIN_ID=0 is the real robot, but this config is for DDS domain 1 '
               '(sim.yaml is for the sim); use g1.yaml')
    with pytest.raises(ValueError, match=re.escape(message)):
        init_channel_factory(sim)
    assert calls == [(1,), (97, 'lo')]


def test_policy_joints_map_to_their_slots(policy):
    assert policy_slots(policy.policy_joints) == tuple(range(0, 13)) + tuple(range(15, 29))
    joints = list(policy.policy_joints)
    with pytest.raises(ValueError, match='not in g1_description'):
        policy_slots(joints[:-1] + ['head_joint'])
    swapped = joints[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match='expected 0-12 and 15-28'):
        policy_slots(swapped)
    with pytest.raises(ValueError, match='expected 0-12 and 15-28'):
        policy_slots(joints[:12] + ['waist_roll_joint'] + joints[13:])


def test_observation_from_lowstate_matches_homie_policy(policy):
    msg = LowStateDefault()
    # a different value in every slot, so a mix-up shows
    for i in range(len(msg.motor_state)):
        msg.motor_state[i].q = 0.01 * i + 0.001
        msg.motor_state[i].dq = -0.1 * i - 0.05
    quat = (math.cos(0.2), 0.6 * math.sin(0.2), 0.8 * math.sin(0.2), 0.0)
    gyro = (0.3, -0.2, 0.1)
    msg.imu_state.quaternion = list(quat)
    msg.imu_state.gyroscope = list(gyro)
    msg.imu_state.rpy = [9.0, 9.0, 9.0]  # not observed
    state = read_state(msg, policy_slots(policy.policy_joints))

    # the same state written down by slot: legs 0-11, waist yaw 12, arms 15-28
    slots = list(range(0, 13)) + list(range(15, 29))
    q = [0.01 * i + 0.001 for i in slots]
    dq = [-0.1 * i - 0.05 for i in slots]
    cmd, height = [0.4, -0.1, 0.3], 0.7
    np.testing.assert_array_equal(
        policy.observation(cmd, height, state.gyro, state.quat_wxyz, state.q, state.dq),
        policy.observation(cmd, height, gyro, quat, q, dq))


def test_gravity_z_matches_homie_policy():
    spec = importlib.util.spec_from_file_location('homie_policy', POLICY_DIR / 'homie_policy.py')
    homie = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(homie)
    rng = np.random.default_rng(0)
    for quat in rng.normal(size=(20, 4)):
        quat /= np.linalg.norm(quat)
        assert gravity_z(quat) == pytest.approx(homie.get_gravity_orientation(quat)[2], abs=1e-12)
    assert gravity_z(_tilted(0.0)) == -1.0
    assert gravity_z(_tilted(60.0)) == pytest.approx(-0.5)


def test_command_filter_clips_then_rate_limits(policy, trained):
    commands = load_config(CONFIG_DIR / 'g1.yaml', trained)['commands']
    f = CommandFilter(policy, commands)
    assert f.cmd.tolist() == [0.0, 0.0, 0.0] and f.height == pytest.approx(0.74)

    # one step moves by at most accel * dt, and the height by height_rate * dt
    cmd, height = f.update([0.4, -0.2, 0.5], 0.5)
    np.testing.assert_allclose(cmd, [0.5 * POLICY_DT, -0.5 * POLICY_DT, 1.0 * POLICY_DT])
    assert height == pytest.approx(0.74 - 0.1 * POLICY_DT)

    # beyond the trained ranges: clipped to them, then to the deployment limits
    for _ in range(300):
        cmd, height = f.update([2.0, -2.0, 2.0], 0.1)
    np.testing.assert_allclose(cmd, [0.5, -0.3, 0.6])
    assert height == pytest.approx(0.24)  # the trained height range
    for _ in range(300):
        cmd, height = f.update([-2.0, 2.0, -2.0], 2.0)
    np.testing.assert_allclose(cmd, [-0.3, 0.3, -0.6])
    assert height == pytest.approx(0.74)
    with pytest.raises(ValueError, match='not finite'):
        f.update([math.nan, 0.0, 0.0], 0.74)


def test_waits_for_steady_upright_state_and_the_arm_controller(policy, trained):
    rig = Rig(policy, trained)
    legs = np.asarray(policy.default_angles) + 0.2
    good = _standing(policy, legs)
    # upright and steady, but the arm controller is not commanding
    assert rig.feed(good, 50, upper=False) == [None] * 50
    # commanding, but tilted 40 deg, then not finite: waiting, not a fault
    assert rig.feed(_standing(policy, legs, quat=_tilted(40.0)), 50) == [None] * 50
    bad = _standing(policy, legs, gyro=(math.nan, 0.0, 0.0))
    assert rig.feed(bad, 50) == [None] * 50
    # gaps above faults.max_state_gap (50 ms) keep it waiting too, and so do
    # samples that reach the node over half faults.max_latency (12.5 ms) after
    # their write, or that far apart: the laptop is not keeping up
    assert rig.feed(good, 5, dt=0.06) == [None] * 5
    assert rig.feed(good, 5, dt=0.03) == [None] * 5
    assert rig.feed(good, 5, age=0.03) == [None] * 5
    assert rig.feed(good, 5, age=0.013) == [None] * 5
    assert rig.controller.phase == WAITING and rig.controller.fault is None
    assert _messages(rig.log) == [
        'waiting to start: no commands from the arm controller on rt/safety/lowcmd_upper_in',
        'waiting to start: the robot is not upright '
        '(gravity z -0.77 in the pelvis frame, needs < -0.9)',
        'waiting to start: rt/lowstate gyro is not finite (at [0])',
        'waiting to start: rt/lowstate is not steady (60 ms between samples, max 50 ms)',
        'waiting to start: rt/lowstate reaches the node late (30 ms, max 12.5 ms to start)',
    ]

    # then STEADY_SECONDS (1 s of ticks) of good samples starts the ramp, from
    # the measured legs; the arm controller must still be commanding
    out = rig.feed(good, SAMPLES_TO_START + 30)
    first = next(i for i, targets in enumerate(out) if targets is not None)
    assert first == SAMPLES_TO_START
    np.testing.assert_allclose(out[first], legs)
    assert rig.controller.phase == RAMPING
    assert not rig.session.inputs, 'the policy must not run during the ramp'


def test_alternating_reasons_to_wait_are_logged_at_most_every_5_s(policy, trained):
    rig = Rig(policy, trained)
    good, tilted = _standing(policy), _standing(policy, quat=_tilted(40.0))
    # two reasons that alternate for 8 s: a tilted robot between upright
    # samples, while the arm controller has not started
    for _ in range(400):
        rig.feed(good, 5, upper=False)
        rig.feed(tilted, 5, upper=False)
    assert rig.controller.phase == WAITING
    kinds = [m.split(' (')[0] for m in _messages(rig.log)]
    assert kinds == ['waiting to start: no commands from the arm controller on '
                     'rt/safety/lowcmd_upper_in', 'waiting to start: the robot is not upright'] * 2


@pytest.mark.parametrize('ticks', [1, 4], ids=['half', 'double'])
def test_a_tick_off_the_wall_clock_keeps_it_waiting(policy, trained, ticks):
    # rt/lowstate's tick counting 1 or 4 per 2 ms sample, not the 2 that
    # tick_seconds (1 ms) means: the policy would run at half or twice its rate
    rig = Rig(policy, trained)
    good = _standing(policy)
    assert rig.feed(good, 4 * SAMPLES_TO_START, ticks=ticks) == [None] * (4 * SAMPLES_TO_START)
    assert rig.controller.phase == WAITING and rig.controller.fault is None
    rate = ticks / 2
    assert _messages(rig.log) == [
        f"waiting to start: rt/lowstate's tick is off the wall clock ({rate:.2f} s of tick "
        'per s, needs 0.90-1.10; is tick_seconds right?)']
    # once the tick is in ms, the next 1 s of ticks starts it (a tilted sample
    # first ends the window that began with the old tick)
    rig.feed(_standing(policy, quat=_tilted(40.0)), 1)
    out = rig.feed(good, SAMPLES_TO_START + 1)
    assert out[:-1] == [None] * SAMPLES_TO_START and out[-1] is not None
    assert rig.controller.phase == RAMPING
    assert _messages(rig.log)[-1].endswith(
        "(rt/lowstate's tick ran at 1.00 s per s of wall clock)")


def test_the_upper_commands_must_be_recent(policy, trained):
    rig = Rig(policy, trained)
    rig.controller.on_upper_cmd(rig.clock.t)
    rig.clock.t += 0.51  # the arm controller's last command was 0.51 s ago
    assert rig.feed(_standing(policy), 100, upper=False) == [None] * 100
    assert rig.controller.phase == WAITING


def test_ramp_then_policy_from_zero_command_and_standing_height(policy, trained):
    rig = Rig(policy, trained, ramp_seconds=0.1)  # 5 policy steps
    legs = np.asarray(policy.default_angles) + np.linspace(-0.3, 0.3, 12)
    state = _standing(policy, legs, gyro=(0.1, -0.2, 0.05))
    rig.walk(state)

    # 6 ramp targets on the line from the measured legs to the default pose,
    # one per policy step (20 ticks), then the first policy step
    assert np.diff(rig.steps).tolist() == [20] * 6
    assert rig.controller.phase == WALKING and len(rig.session.inputs) == 1
    assert [m for m in _messages(rig.log) if not m.startswith('waiting')] == [
        "moving the legs from where they are to the default pose over 0.1 s (rt/lowstate's "
        'tick ran at 1.00 s per s of wall clock)',
        'walking: policy started at zero velocity and 0.74 m height',
    ]

    # the policy starts from an empty history, zero velocity and 0.74 m, and
    # observes the state as HomiePolicy.observation does
    obs = rig.session.inputs[0].reshape(6, -1)
    assert not obs[:5].any(), 'the history must start empty'
    reference = load_policy(str(POLICY_DIR))
    expected = reference.observation(
        [0.0, 0.0, 0.0], 0.74, state.gyro, state.quat_wxyz, state.q, state.dq)
    np.testing.assert_array_equal(obs[5], expected)
    assert rig.controller.command[1] == pytest.approx(0.74)


def test_ramp_is_linear_to_the_default_pose(policy, trained):
    rig = Rig(policy, trained, ramp_seconds=0.1)
    legs = np.asarray(policy.default_angles) + np.linspace(-0.3, 0.3, 12)
    out = [t for t in rig.feed(_standing(policy, legs), SAMPLES_TO_START + 200) if t is not None]
    ramp = np.array(out[:6])
    alpha = np.linspace(0.0, 1.0, 6)[:, None]
    default = np.asarray(policy.default_angles)
    np.testing.assert_allclose(ramp, legs + alpha * (default - legs), atol=1e-12)
    # and the policy after it (zero actions: the default pose)
    np.testing.assert_allclose(out[6], policy.default_angles, atol=1e-6)


def test_cmd_vel_is_rate_limited_and_times_out(policy, trained):
    rig = Rig(policy, trained)
    state = _standing(policy)
    rig.walk(state)
    rig.controller.set_cmd_vel(0.4, 0.0, 0.0, rig.clock.t)
    rig.feed(state, 20 * SAMPLES_PER_STEP)  # 0.4 s: 20 steps of +0.01
    vx = rig.controller.command[0][0]
    assert vx == pytest.approx(0.2)
    # the last policy input carries the command, scaled by cmd_scale (2)
    assert rig.session.inputs[-1][0, 5 * 76] == pytest.approx(2 * vx, abs=1e-6)
    # older than commands.timeout (0.5 s): heads back to zero
    rig.feed(state, 20 * SAMPLES_PER_STEP)
    assert rig.controller.command[0][0] < vx
    rig.feed(state, 50 * SAMPLES_PER_STEP)
    assert rig.controller.command[0].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    # a /cmd_vel that is not finite means stop; a bad height is ignored
    rig.controller.set_cmd_vel(math.nan, 0.0, 0.0, rig.clock.t)
    rig.controller.set_height(math.inf)
    rig.controller.set_height(0.6)
    rig.feed(state, 10 * SAMPLES_PER_STEP)
    cmd, height = rig.controller.command
    assert cmd.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert height == pytest.approx(0.74 - 10 * 0.002)
    assert _messages(rig.log, 'warn') == [
        '/cmd_vel is not finite ([nan, 0.0, 0.0]): stopping',
        'ignoring a /locomotion/height that is not finite (inf)',
    ]


def _changed(**fields):
    '''Inject one sample with these State fields changed'''
    def inject(rig, state):
        changes = {name: np.asarray(value, dtype=np.float64) for name, value in fields.items()}
        rig.feed(state._replace(**changes), 1)
    return inject


def _knee_not_finite(rig, state):
    rig.feed(state._replace(q=np.where(np.arange(27) == 3, math.inf, state.q)), 1)


def _session_runs(make_hook):
    '''Inject a policy step whose ONNX session runs make_hook(rig)()'''
    def inject(rig, state):
        rig.session.hook = make_hook(rig)
        rig.feed(state, SAMPLES_PER_STEP)
    return inject


def _slow(rig):
    def hook():
        rig.clock.t += 0.030
        return np.zeros(12)
    return hook


def _failing(rig):
    def hook():
        raise RuntimeError('onnxruntime failed')
    return hook


# each fault injected while walking: (name, how, the fault's start)
FAULTS = [
    ('gap', lambda rig, state: rig.feed(state, 1, dt=0.06),
     'rt/lowstate stale: 60 ms between samples (> 50 ms)'),
    ('silence', lambda rig, state: rig.controller.check_stale(rig.clock.t + 0.051),
     'rt/lowstate stale: no sample for 51 ms (> 50 ms)'),
    ('gyro', _changed(gyro=[0.0, math.nan, 0.0]), 'rt/lowstate gyro is not finite (at [1])'),
    ('q', _knee_not_finite, "rt/lowstate q is not finite (at ['left_knee_joint'])"),
    ('quaternion', _changed(quat_wxyz=[0.5, 0.0, 0.0, 0.0]),
     'the rt/lowstate IMU quaternion is not a unit quaternion (norm 0.500)'),
    ('fall', _changed(quat_wxyz=_tilted(70.0)),
     'fell: gravity z -0.34 in the pelvis frame (> -0.5)'),
    ('tick back', lambda rig, state: rig.feed(state, 1, ticks=-10),
     'rt/lowstate tick went back from 2142 to 2132'),
    ('late', _session_runs(_slow),
     'command late: 30.0 ms after its rt/lowstate sample (> 25 ms)'),
    ('policy error', _session_runs(_failing), 'policy step failed: onnxruntime failed'),
    ('policy output', _session_runs(lambda rig: lambda: np.full(12, math.nan)),
     'policy output is not 12 finite targets'),
]


@pytest.mark.parametrize('name, inject, reason', FAULTS, ids=[f[0] for f in FAULTS])
def test_faults_latch_silent(policy, trained, name, inject, reason):
    rig = Rig(policy, trained)
    state = _standing(policy)
    rig.walk(state)
    assert [t is not None for t in rig.feed(state, SAMPLES_PER_STEP)].count(True) == 1
    inject(rig, state)
    assert rig.controller.phase == FAULTED and rig.controller.fault.startswith(reason)
    # silent for good, whatever comes next
    assert rig.feed(state, 20 * SAMPLES_PER_STEP) == [None] * (20 * SAMPLES_PER_STEP)
    errors = _messages(rig.log, 'error')
    assert len(errors) == 1 and errors[0].startswith(f'fault: {reason}')
    assert "g1_safety_layer's leg watchdog estops next. Restart the node to clear" in errors[0]


def test_faults_during_the_ramp_latch_too(policy, trained):
    rig = Rig(policy, trained, ramp_seconds=1.0)
    state = _standing(policy, np.asarray(policy.default_angles) + 0.2)
    rig.feed(state, SAMPLES_TO_START + 100)
    assert rig.controller.phase == RAMPING
    assert rig.feed(state._replace(quat_wxyz=np.array(_tilted(70.0))), 1) == [None]
    assert rig.controller.phase == FAULTED and rig.controller.fault.startswith('fell')
    assert rig.feed(state, 100) == [None] * 100
    assert not rig.session.inputs


def test_tick_delta_wraps():
    assert tick_delta(1020, 1000) == 20 and tick_delta(1000, 1020) == -20
    assert tick_delta(5, TICK_MODULO - 15) == 20
    assert tick_delta(TICK_MODULO - 15, 5) == -20
    assert tick_delta(7, 7) == 0


def _walking(policy, trained, **kwargs):
    rig = Rig(policy, trained, **kwargs)
    rig.walk(_standing(policy))
    rig.steps.clear()
    return rig


@pytest.mark.parametrize('spacing', [
    [2],  # every 500 Hz sample
    [6],  # every third sample
    [2, 4, 2, 8, 6, 2, 12, 2, 16, 4],  # irregular, as when the node falls behind
], ids=['every sample', 'every third', 'irregular'])
def test_policy_steps_on_the_tick_grid_whatever_samples_arrive(policy, trained, spacing):
    rig = _walking(policy, trained)
    state = _standing(policy)
    grid = rig.controller._due_tick
    for i in range(3000):
        ticks = spacing[i % len(spacing)]
        rig.feed(state, 1, dt=ticks * TICK_S, ticks=ticks)
    steps = np.array(rig.steps) - grid
    gaps = np.diff(steps)
    # never closer than half a period
    assert gaps.min() >= 10, gaps.min()
    # each step is at the first sample at or past a grid tick (multiples of
    # 20 from the walk's first step), and a grid tick is skipped only when the
    # step before it came more than half a period late
    late = steps % 20
    assert np.all(late < max(spacing)), late.max()
    periods = np.diff(steps // 20)
    assert np.all((periods == 1) | (late[:-1] > 10)), list(zip(periods, late[:-1]))
    if max(spacing) <= 10:
        # no step is that late: exactly 50 Hz of robot time on average
        assert abs(gaps.mean() - 20.0) < 0.05, gaps.mean()
    if spacing == [2]:
        assert set(gaps.tolist()) == {20}
    assert rig.controller.fault is None


def test_after_a_backlog_the_newest_sample_steps_once(policy, trained):
    # the driver hands over only the newest sample: one 46 ms of ticks after the
    # last (within faults.max_state_gap), then samples 2 ms apart again
    rig = _walking(policy, trained)
    state = _standing(policy)
    rig.feed(state, SAMPLES_PER_STEP)
    last = rig.steps[-1]
    out = rig.feed(state, 1, dt=0.046)
    assert out[0] is not None and rig.steps[-1] == last + 46, 'one step, on the newest sample'
    rig.feed(state, 3 * SAMPLES_PER_STEP)
    # back on the grid: the next steps at last + 60, + 80, ...
    assert rig.steps[-3:] == [last + 60, last + 80, last + 100]
    assert rig.controller.fault is None


def test_samples_whose_tick_did_not_advance_are_ignored(policy, trained):
    rig = _walking(policy, trained)
    state = _standing(policy)
    inputs = len(rig.session.inputs)
    # the same tick again and again for 60 ms: no step, and not fresh
    assert rig.feed(state, 30, ticks=0) == [None] * 30
    assert len(rig.session.inputs) == inputs
    rig.controller.check_stale(rig.clock.t)
    assert rig.controller.fault == 'rt/lowstate stale: no sample for 60 ms (> 50 ms)'


def test_the_tick_wraps_while_walking(policy, trained):
    rig = _walking(policy, trained, tick=TICK_MODULO - 1500)
    rig.feed(_standing(policy), 50 * SAMPLES_PER_STEP)
    assert rig.steps[0] > TICK_MODULO - 1500 and rig.steps[-1] < 1000, 'it wrapped'
    assert {tick_delta(b, a) for a, b in zip(rig.steps, rig.steps[1:])} == {20}
    assert rig.controller.fault is None


def test_latency_limit_and_timing_log(policy, trained):
    rig = _walking(policy, trained)
    state = _standing(policy)
    # the walk logs its timing every STATS_SECONDS
    rig.feed(state, int((STATS_SECONDS + 0.5) / SAMPLE_DT))
    timing = [m for m in _messages(rig.log, 'info') if m.startswith('timing')]
    assert len(timing) == 1, timing
    assert re.fullmatch(r'timing over the last 10 s: 50[0-2] commands, latency median 0\.0 ms, '
                        r'max 0\.0 ms \(limit 25 ms\); steps 20-20 ms of rt/lowstate time apart, '
                        r'mean 20\.0 ms', timing[0]), timing[0]
    # at the limit is fine, over it a fault, and nothing more is sent
    controller = rig.controller
    assert controller.latency_ok(0.025)
    assert not controller.latency_ok(0.0251)
    assert controller.phase == FAULTED
    assert controller.fault == 'command late: 25.1 ms after its rt/lowstate sample (> 25 ms)'
    assert not controller.latency_ok(0.001)
    assert rig.feed(state, 100) == [None] * 100


def test_source_clock_measures_from_the_write():
    clock = SourceClock()
    offset, fastest = 1000.0, 0.0004  # the clocks' offset and the fastest delivery
    # samples 2 ms apart, delivered in 0.4-0.9 ms: the write is put at the
    # source time plus offset plus the fastest delivery
    for i in range(500):
        source = 50.0 + 0.002 * i
        received = source + offset + fastest + 0.0005 * (i % 2)
        written = clock.written(source, received)
        assert written == pytest.approx(source + offset + fastest, abs=1e-9)
    # a sample delivered 40 ms late is measured from its write, not its arrival
    source = 51.0
    written = clock.written(source, source + offset + fastest + 0.040)
    assert (source + offset + fastest + 0.040) - written == pytest.approx(0.040, abs=1e-9)
    # an early outlier (the clocks' offset changed) is forgotten after ~2 s
    clock.written(52.0, 52.0 + offset + fastest - 0.010)
    for i in range(1, 1501):
        source = 52.0 + 0.002 * i
        written = clock.written(source, source + offset + fastest)
    assert written == pytest.approx(source + offset + fastest, abs=1e-9)


def test_policy_session_is_single_threaded_with_the_same_actions(policy):
    import onnxruntime as ort
    options = policy.session.get_session_options()
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (1, 1)
    # HomiePolicy's own session, with onnxruntime's default threads
    default = ort.InferenceSession(policy.config['policy_path'],
                                   providers=['CPUExecutionProvider'])
    size = policy.session.get_inputs()[0].shape[-1]
    obs = np.random.default_rng(0).normal(0.0, 1.0, (300, 1, size)).astype(np.float32)
    for x in obs:
        ours = policy.session.run(None, {policy.input_name: x})[0]
        theirs = default.run(None, {policy.input_name: x})[0]
        np.testing.assert_array_equal(ours, theirs)
