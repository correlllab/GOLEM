'''
The walking node's DDS side on DDS domain 97 (no robot): in this process a fake G1 on
rt/lowstate, a fake arm controller on the upper input and the node's controller and driver,
on Unitree SDK channels; beside it golem_safety's installed safety_node as a subprocess, with
robot g1 on a copy of its split.yaml (split mode). ROS topic /x is DDS rt/x with the same
unitree_hg types, so the SDK channels here and the safety node's ROS topics meet
'''

import itertools
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from g1_locomotion.config import check_config
from g1_locomotion.driver import LocomotionDriver
from g1_locomotion.locomotion import (
    FAULTED,
    WAITING,
    WALKING,
    LocomotionController,
    load_policy,
    tick_delta,
)

# the package's policy/ (the containers also mount it at /home/code/g1_locomotion)
POLICY_DIR = Path(__file__).resolve().parents[1] / 'policy'
CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
# This one process runs the fakes, the recorders and the node, and the GIL
# delays rt/lowstate by tens of ms at times (up to 90 ms seen), which the real
# node, in a process of its own, does not see. So these tests relax g1.yaml's
# 50 ms gap and 25 ms latency, and the tick rate the start accepts (a late
# sample at either end of its 1 s window moves it by as much); test_locomotion.py
# checks the real values. golem_safety's quiet-rt/lowstate report
# (stale.state_seconds, 0.1 s in split.yaml) gets MAX_STATE_GAP too, except where
# a test keeps split.yaml's value
MAX_STATE_GAP = 0.25
MAX_LATENCY = 0.25
TICK_RATE = [0.5, 2.0]
# rt/lowstate samples per policy step when every one arrives
SAMPLES_PER_STEP = 10
LEGS = range(0, 12)
SHOULDER = 15  # left shoulder pitch, an arm slot
# slack for thread scheduling when timing a report of golem_safety's
LATE_S = 0.3
# the events by which golem_safety latches its estop or reports a quiet input or a
# quiet /lowstate: none of them may appear while the node walks
ALARMS = ('estop_command_written', 'input_quiet', 'state_quiet')
# rcutils' default console format: [LEVEL] [seconds] [logger]: message
LOG_LINE = re.compile(r'^\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[([0-9.]+)\] \[[^\]]*\]: (.*)$')
_factory = []
_runs = itertools.count()


def _init_dds():
    # the Unitree channel factory is process-wide; initialize it once
    if not _factory:
        ChannelFactoryInitialize(TEST_DOMAIN)
        _factory.append(True)


def _tilted(degrees):
    '''Pelvis quaternion (w, x, y, z) rolled by `degrees` about x'''
    half = math.radians(degrees) / 2
    return [math.cos(half), math.sin(half), 0.0, 0.0]


class FakeG1:
    '''
    rt/lowstate at 500 Hz from a G1 standing still, its tick in ms (+2 a
    sample); set() changes it, pause() stops it
    '''

    def __init__(self, topic, legs):
        self.lock = threading.Lock()
        self.state = LowStateDefault()
        self.state.mode_machine = 5
        self.state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        for i, q in zip(LEGS, legs):
            self.state.motor_state[i].q = float(q)
        self.samples = 0
        self._resume = 0.0
        self._stop = threading.Event()
        self._pub = ChannelPublisher(topic, LowState_)
        self._pub.Init()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set(self, quat=None, gyro=None):
        with self.lock:
            if quat is not None:
                self.state.imu_state.quaternion = list(quat)
            if gyro is not None:
                self.state.imu_state.gyroscope = list(gyro)

    def pause(self, seconds):
        with self.lock:
            self._resume = time.monotonic() + seconds

    def _loop(self):
        next_time = time.monotonic()
        while not self._stop.is_set():
            with self.lock:
                if time.monotonic() >= self._resume:
                    self.state.tick += 2
                    self._pub.Write(self.state)
                    self.samples += 1
            next_time += 0.002
            time.sleep(max(0.0, next_time - time.monotonic()))

    def close(self):
        self._stop.set()
        self._thread.join()
        self._pub.Close()


class FakeArms:
    '''The arm controller: waist + arms held at 0 (kp 60) on the upper input at 100 Hz'''

    def __init__(self, topic):
        self.cmd = LowCmdDefault()
        for i in range(12, 29):
            self.cmd.motor_cmd[i].mode = 1
            self.cmd.motor_cmd[i].kp = 60.0
            self.cmd.motor_cmd[i].kd = 1.5
        self._stop = threading.Event()
        self._pub = ChannelPublisher(topic, LowCmd_)
        self._pub.Init()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            self._pub.Write(self.cmd)
            time.sleep(0.01)

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._pub.Close()


class Recorder:
    '''Every LowCmd_ on a topic, with its receive time (monotonic)'''

    def __init__(self, topic):
        self.msgs = []
        self._sub = ChannelSubscriber(topic, LowCmd_)
        self._sub.Init(lambda m: self.msgs.append((time.monotonic(), m)), 50)

    def close(self):
        self._sub.Close()


class SafetyNode:
    '''
    golem_safety's installed safety_node on TEST_DOMAIN, robot g1 on the config
    at `config`, sim:=true (so it refuses domain 0) and no hardware estop; a
    thread collects its log lines
    '''

    def __init__(self, config):
        prefix = Path(get_package_prefix('golem_safety'))
        executable = prefix / 'lib' / 'golem_safety' / 'safety_node'
        env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN), RCUTILS_COLORIZED_OUTPUT='0')
        env.pop('RCUTILS_CONSOLE_OUTPUT_FORMAT', None)
        env.pop('RCUTILS_LOGGING_USE_STDOUT', None)
        self.lines = []
        self.exit_code = None
        self.proc = subprocess.Popen(
            [str(executable), '--ros-args', '-p', 'robot:=g1', '-p', f'config:={config}',
             '-p', 'sim:=true', '-p', 'use_estop:=false'],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def output(self):
        return '\n'.join(self.lines)

    def events(self, names=None):
        '''The JSON events the node logged, in order; only those called one of names, if given'''
        events = []
        for line in list(self.lines):
            match = LOG_LINE.match(line)
            if match and match.group(3).startswith('{'):
                event = json.loads(match.group(3))
                if names is None or event['event'] in names:
                    events.append(event)
        return events

    def wait_for(self, name, timeout=15.0):
        deadline = time.monotonic() + timeout
        while not self.events((name,)):
            assert self.proc.poll() is None, f'safety_node exited:\n{self.output()}'
            assert time.monotonic() < deadline, f'no {name} within {timeout} s:\n{self.output()}'
            time.sleep(0.02)

    def stop(self):
        '''SIGINT, as Ctrl-C; SIGKILL if it hangs. Keeps the exit code'''
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._reader.join(timeout=5.0)
        self.exit_code = self.proc.returncode


class Rig:
    '''
    What _rig() starts; log holds the node's (time, level, message), fed
    every sample the driver gave the controller: (tick, time, whether it
    returned targets), and latencies every command's (time, latency) as the
    driver measured it
    '''


@contextmanager
def _rig(legs, ramp_seconds=0.2, state_seconds=MAX_STATE_GAP):
    '''
    The node's controller and driver (g1.yaml, with a short ramp and the
    relaxed timing above), golem_safety's safety_node on a copy of its
    split.yaml (logging off; stale.state_seconds at state_seconds s, or
    split.yaml's for None), a fake G1 and a fake arm controller (not started),
    on topics of their own, so no channel of another test can match this run's.
    The safety node is stopped first, so that the rest of the teardown adds
    nothing to its log
    '''
    _init_dds()
    # a ROS name token must not start with a digit
    suffix = f'/test_locomotion_dds/run{next(_runs)}'
    split = Path(get_package_share_directory('golem_safety')) / 'config' / 'g1' / 'split.yaml'
    safety = yaml.safe_load(split.read_text())
    safety['logging']['enabled'] = False
    safety['topics'] = {key: name + suffix for key, name in safety['topics'].items()}
    if state_seconds is not None:
        safety['stale']['state_seconds'] = state_seconds
    with open(CONFIG_DIR / 'g1.yaml') as f:
        raw = yaml.safe_load(f)
    raw['topics'] = {key: name + suffix for key, name in raw['topics'].items()}
    raw['startup'] = {'ramp_seconds': ramp_seconds, 'tick_rate': TICK_RATE}
    raw['faults'] = {'max_state_gap': MAX_STATE_GAP, 'max_latency': MAX_LATENCY}

    rig = Rig()
    rig.safety = safety
    rig.policy = load_policy(str(POLICY_DIR))
    rig.config = check_config(raw, rig.policy.config['cmd_range'])
    # the node's DDS topics are the safety node's ROS topics
    for node_key, safety_key in (('low_cmd', 'low_cmd_lower_in'),
                                 ('upper_cmd', 'low_cmd_upper_in'),
                                 ('low_state', 'low_state')):
        assert rig.config['topics'][node_key] == 'rt' + safety['topics'][safety_key]
    rig.log = []
    controller = LocomotionController(
        rig.policy, rig.config,
        log=lambda level, message: rig.log.append((time.monotonic(), level, message)))
    rig.fed = []
    on_lowstate = controller.on_lowstate

    def fed(state, tick, now, age=0.0):
        targets = on_lowstate(state, tick, now, age)
        rig.fed.append((tick, now, targets is not None))
        return targets

    controller.on_lowstate = fed
    rig.latencies = []
    latency_ok = controller.latency_ok

    def timed(latency):
        rig.latencies.append((time.monotonic(), latency))
        return latency_ok(latency)

    controller.latency_ok = timed
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / 'g1' / 'split.yaml'
        config.parent.mkdir()
        config.write_text(yaml.safe_dump(safety))
        rig.safety_node = SafetyNode(config)
        # what started, closed in reverse order after the safety node, whatever
        # fails while starting the rest
        started = []
        try:
            rig.lowcmd = Recorder('rt' + safety['topics']['low_cmd_out'])
            started.append(rig.lowcmd.close)
            rig.lower = Recorder(rig.config['topics']['low_cmd'])
            started.append(rig.lower.close)
            rig.robot = FakeG1(rig.config['topics']['low_state'], legs)
            started.append(rig.robot.close)
            rig.arms = FakeArms(rig.config['topics']['upper_cmd'])
            started.append(rig.arms.close)
            rig.driver = LocomotionDriver(controller, rig.config['topics'], TEST_DOMAIN)
            started.append(rig.driver.shutdown)
            rig.safety_node.wait_for('relay_started')
            # discovery between this process and the safety node's
            time.sleep(0.5)
            yield rig
        finally:
            rig.safety_node.stop()
            for close in reversed(started):
                close()


def _wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.01)


def _legs(msg):
    return np.array([msg.motor_cmd[i].q for i in LEGS])


def _is_off(motors):
    return all(m.mode == 0 and m.q == m.dq == m.tau == m.kp == m.kd == 0.0 for m in motors)


def _is_zeroed(msg):
    '''Every motor off: golem_safety's zero-gain stream, and its estop command'''
    return _is_off(msg.motor_cmd[:29])


def _alarms(rig):
    '''golem_safety's estops and quiet reports (ALARMS) of the run; the node must have exited 0'''
    assert rig.safety_node.exit_code == 0, rig.safety_node.output()
    return rig.safety_node.events(ALARMS)


def test_silent_until_upright_state_and_arms_then_ramp_and_walk():
    default = np.asarray(load_policy(str(POLICY_DIR)).default_angles, dtype=np.float64)
    legs = default + np.linspace(-0.2, 0.2, 12)
    with _rig(legs, ramp_seconds=0.2) as rig:
        kp, kd = rig.policy.gains(rig.policy.leg_joints)
        # upright rt/lowstate but no arm controller: the node is silent, and
        # golem_safety streams zero gains on every motor
        time.sleep(0.4)
        assert rig.lower.msgs == []
        assert rig.lowcmd.msgs and all(_is_zeroed(m) for _, m in rig.lowcmd.msgs)
        # the arm controller commands, but the robot is tilted 40 deg: the node
        # stays silent, and golem_safety relays the arms with the legs still off
        rig.robot.set(quat=_tilted(40.0))
        rig.arms.start()
        time.sleep(0.4)
        assert rig.lower.msgs == []
        assert rig.lowcmd.msgs[-1][1].motor_cmd[SHOULDER].kp == 60.0
        assert all(_is_off(m.motor_cmd[:12]) for _, m in rig.lowcmd.msgs)
        assert rig.driver.phase == WAITING

        # upright: the ramp (0.2 s, 11 targets), then the policy
        rig.robot.set(quat=[1.0, 0.0, 0.0, 0.0])
        _wait_for(lambda: rig.driver.phase == WALKING)
        time.sleep(0.3)
        start_samples, start = rig.robot.samples, time.monotonic()
        time.sleep(1.0)
        samples, end = rig.robot.samples - start_samples, time.monotonic()
        lower = list(rig.lower.msgs)
        lowcmd = list(rig.lowcmd.msgs)
        fed = list(rig.fed)
        assert rig.driver.fault is None, rig.driver.fault

    # the ramp starts at the measured legs and moves each one monotonically to
    # the default pose, one target per policy step
    ramp = np.array([_legs(m) for _, m in lower[:11]])
    np.testing.assert_allclose(ramp[0], legs, atol=1e-6)
    np.testing.assert_allclose(ramp[-1], default, atol=1e-6)
    toward = np.sign(default - legs)
    assert np.all(np.diff(ramp, axis=0) * toward >= -1e-6), 'must move toward the defaults'
    steps = np.diff([t for t, _ in lower[:11]])
    assert 0.01 < np.median(steps) < 0.03, f'one target per policy step, not {np.median(steps)} s'

    # every leg command: slots 0-11 only, mode 1, dq = tau = 0, the training gains
    for _, msg in lower:
        for i in LEGS:
            m = msg.motor_cmd[i]
            assert (m.mode, m.dq, m.tau) == (1, 0.0, 0.0)
            assert m.kp == pytest.approx(kp[i]) and m.kd == pytest.approx(kd[i])
        assert _is_off(msg.motor_cmd[12:])

    # one command per 20 ms of tick: 50 Hz on a 500 Hz robot, one per 10 samples
    walking = [t for t, _ in lower if start <= t < end]
    assert abs(len(walking) - samples / SAMPLES_PER_STEP) <= 3, (len(walking), samples)
    rate = len(walking) / (end - start)
    assert 40.0 < rate < 55.0, f'{rate:.1f} Hz'
    steps = [tick for tick, t, stepped in fed if stepped and start <= t < end]
    gaps = [tick_delta(b, a) for a, b in zip(steps, steps[1:])]
    assert min(gaps) >= 10 and abs(np.mean(gaps) - 20.0) < 1.0, gaps

    # golem_safety engaged the arm controller's input first and the node's on
    # its first command (the measured legs), each on its own; the legs were
    # off until then, and from then on carry the node's legs with the training
    # gains beside the arm controller's arms. Every command is stamped with the
    # G1 profile's mode_pr 0 and mode_machine 5
    assert [e['source'] for e in rig.safety_node.events(('input_engaged',))] == [
        'split_mode_upper', 'split_mode_lower']
    first = next(i for i, (_, m) in enumerate(lowcmd) if m.motor_cmd[LEGS[0]].kp > 0.0)
    assert all(_is_off(m.motor_cmd[:12]) for _, m in lowcmd[:first])
    assert all((m.mode_pr, m.mode_machine) == (0, 5) for _, m in lowcmd[:first])
    np.testing.assert_allclose(_legs(lowcmd[first][1]), legs, atol=1e-6)
    sent = {tuple(np.round(_legs(m), 6)) for _, m in lower}
    for _, msg in lowcmd[first:]:
        assert (msg.mode_pr, msg.mode_machine) == (0, 5)
        assert tuple(np.round(_legs(msg), 6)) in sent
        assert all(msg.motor_cmd[i].mode == 1 and msg.motor_cmd[i].kp == pytest.approx(kp[i]) and
                   msg.motor_cmd[i].kd == pytest.approx(kd[i]) for i in LEGS)
        assert msg.motor_cmd[SHOULDER].kp == 60.0
    assert _alarms(rig) == []
    assert [m for _, level, m in rig.log if level == 'error'] == []


def _nan_gyro(rig):
    rig.robot.set(gyro=[math.nan, 0.0, 0.0])


def _fall(rig):
    rig.robot.set(quat=_tilted(70.0))


def _stale(rig):
    # no rt/lowstate for longer than the node allows, then it comes back
    rig.robot.pause(MAX_STATE_GAP + 0.15)


class SlowSession:
    '''An ONNX session whose runs take longer than faults.max_latency'''

    def __init__(self, session):
        self._session = session

    def run(self, output_names, feed):
        time.sleep(MAX_LATENCY + 0.05)
        return self._session.run(output_names, feed)


def _slow_policy(rig):
    rig.policy.session = SlowSession(rig.policy.session)


@pytest.mark.parametrize('inject, reason', [
    (_stale, 'rt/lowstate stale'),
    (_nan_gyro, 'rt/lowstate gyro is not finite'),
    (_fall, 'fell: gravity z'),
    (_slow_policy, 'command late: 3'),
], ids=['stale', 'not finite', 'fallen', 'late'])
def test_bad_state_silences_the_node_and_golem_safety_holds_its_last_command(inject, reason):
    default = load_policy(str(POLICY_DIR)).default_angles
    # a stale rt/lowstate meets split.yaml's own quiet-rt/lowstate time; the
    # other faults leave rt/lowstate flowing
    state_seconds = None if inject is _stale else MAX_STATE_GAP
    with _rig(default, ramp_seconds=0.1, state_seconds=state_seconds) as rig:
        lower_quiet = rig.safety['stale']['lower_seconds']
        state_quiet = rig.safety['stale']['state_seconds']
        kp, kd = rig.policy.gains(rig.policy.leg_joints)
        rig.arms.start()
        _wait_for(lambda: rig.driver.phase == WALKING)
        time.sleep(0.3)
        inject_time = time.monotonic()
        inject(rig)
        time.sleep(MAX_STATE_GAP + lower_quiet + LATE_S)
        end = time.monotonic()
        lower = list(rig.lower.msgs)
        lowcmd = list(rig.lowcmd.msgs)
        fault = rig.driver.fault
        assert rig.driver.phase == FAULTED and fault.startswith(reason), fault

    # the node latched once, soon after the fault, and sent nothing after that
    errors = [(t, m) for t, level, m in rig.log if level == 'error']
    assert len(errors) == 1 and errors[0][1].startswith(f'fault: {reason}'), errors
    fault_time = errors[0][0]
    assert inject_time < fault_time < inject_time + MAX_STATE_GAP + 0.15
    last_lower = lower[-1][0]
    assert last_lower < fault_time + 0.05, 'nothing may be sent after the fault'
    # never a zero-gain or damping command from the node itself
    assert all(msg.motor_cmd[i].kp > 0.0 for _, msg in lower for i in LEGS)

    # golem_safety holds a quiet input: from the arm controller's first command
    # on (before it, the zero-gain stream), it went on relaying the node's last
    # leg command, with the training gains, to the end, and never the estop
    # command
    arms = next(i for i, (_, msg) in enumerate(lowcmd) if msg.motor_cmd[SHOULDER].kp > 0.0)
    assert all(_is_zeroed(msg) for _, msg in lowcmd[:arms])
    assert not any(_is_zeroed(msg) for _, msg in lowcmd[arms:])
    held = [(t, msg) for t, msg in lowcmd if t > last_lower + 0.05]
    assert held, 'golem_safety must keep relaying'
    last_legs = _legs(lower[-1][1])
    for _, msg in held:
        np.testing.assert_allclose(_legs(msg), last_legs, atol=1e-6)
        assert all(msg.motor_cmd[i].mode == 1 and msg.motor_cmd[i].kp == pytest.approx(kp[i]) and
                   msg.motor_cmd[i].kd == pytest.approx(kd[i]) for i in LEGS)
    assert held[0][0] < last_lower + 0.15 and held[-1][0] > end - 0.1, (
        last_lower, held[0][0], held[-1][0], end)
    rate = (len(held) - 1) / (held[-1][0] - held[0][0])
    assert rate > 250.0, f'{rate:.0f} Hz of held commands; golem_safety relays at 500 Hz'

    # and reported the quiet leg input once, lower_quiet s after the node's last
    # command (its age is rounded to 4 decimals), without an estop and with no
    # command to resume it
    events = rig.safety_node.events()
    quiet = [e for e in events if e['event'] == 'input_quiet']
    assert [(e['source'], e['topic'], e['seconds']) for e in quiet] == [
        ('split_mode_lower', rig.safety['topics']['low_cmd_lower_in'], lower_quiet)], quiet
    assert lower_quiet <= quiet[0]['age'] < lower_quiet + LATE_S, quiet
    assert [e for e in events if e['event'] == 'input_resumed'] == []
    assert [e for e in events if e['event'].startswith('estop')] == []
    assert rig.safety_node.exit_code == 0, rig.safety_node.output()
    # a stale rt/lowstate is reported too, and its return; the relay never waited for it
    states = [(e['event'], e['topic']) for e in events
              if e['event'] in ('state_quiet', 'state_resumed')]
    if inject is _stale:
        assert states == [('state_quiet', rig.safety['topics']['low_state']),
                          ('state_resumed', rig.safety['topics']['low_state'])], states
        (reported,) = [e for e in events if e['event'] == 'state_quiet']
        assert reported['seconds'] == state_quiet, reported
        assert state_quiet <= reported['age'] < state_quiet + LATE_S, reported
    else:
        assert states == [], states


def test_a_backlog_is_dropped_and_the_cadence_kept():
    default = load_policy(str(POLICY_DIR)).default_angles
    with _rig(default, ramp_seconds=0.1) as rig:
        rig.arms.start()
        _wait_for(lambda: rig.driver.phase == WALKING)
        time.sleep(0.3)
        # the worker is held up for 120 ms while about 60 samples arrive
        # (golem_safety reports a leg input quiet for 200 ms, its
        # stale.lower_seconds)
        with rig.driver._lock:
            before = rig.robot.state.tick
            time.sleep(0.12)
            newest = rig.robot.state.tick
            release = time.monotonic()
        time.sleep(0.5)
        fed = list(rig.fed)
        lower = list(rig.lower.msgs)
        latencies = list(rig.latencies)
        fault = rig.driver.fault
    assert fault is None, fault
    assert tick_delta(newest, before) >= 50, (before, newest)

    # none of the samples that arrived while it was held reached the
    # controller: it went on from the newest
    ticks = [tick for tick, _, _ in fed]
    assert not [t for t in ticks if 0 < tick_delta(t, before) < tick_delta(newest, before) - 2]
    after = [(tick, stepped) for tick, t, stepped in fed if t >= release]
    assert -2 <= tick_delta(after[0][0], newest) <= 10, (after[0], newest)
    # and the policy stepped on it at once: one command, not a burst of the
    # six the backlog held (the next step is at least 10 ms of tick later),
    # answering a sample written a few ms before
    assert after[0][1], 'the policy must step on the newest sample'
    burst = [t for t, _ in lower if release <= t < release + 0.008]
    assert len(burst) <= 1, burst
    first_latency = next(latency for t, latency in latencies if t >= release)
    assert first_latency < 0.02, first_latency
    # 50 Hz of tick before and after
    steps = [tick for tick, _, stepped in fed if stepped]
    gaps = [tick_delta(b, a) for a, b in zip(steps, steps[1:])]
    assert min(gaps) >= 10, gaps
    assert sorted(gaps)[len(gaps) // 2] == 20, gaps
    assert [m for _, level, m in rig.log if level == 'error'] == []
    assert _alarms(rig) == []


def test_the_latency_counts_the_time_a_sample_waited_for_the_node():
    default = load_policy(str(POLICY_DIR)).default_angles
    with _rig(default, ramp_seconds=0.1) as rig:
        rig.arms.start()
        _wait_for(lambda: rig.driver.phase == WALKING)
        time.sleep(0.3)
        # the worker is held up, and rt/lowstate stops 30 ms in for 150 ms:
        # when the worker goes on, the newest sample is the one written 90 ms
        # before (within the relaxed gap and golem_safety's quiet-rt/lowstate
        # time, and its 200 ms quiet-leg time)
        with rig.driver._lock:
            time.sleep(0.03)
            rig.robot.pause(0.15)
            paused = time.monotonic()
            time.sleep(0.09)
            release = time.monotonic()
        time.sleep(0.4)
        latencies = list(rig.latencies)
        fault = rig.driver.fault
    assert fault is None, fault
    # the command answering it is measured from the sample's DDS write, so
    # it counts the 90 ms the sample waited, not just the node's own work
    waited = release - paused
    first = next(latency for t, latency in latencies if t >= release)
    assert waited - 0.01 < first < waited + 0.1, (first, waited)
    # before and after, well under that: a median of 1-4 ms, up to 7 ms seen
    # when the host was loaded (this process's threads share the GIL)
    others = [latency for t, latency in latencies if not release <= t < release + 0.05]
    assert sorted(others)[len(others) // 2] < 0.02, sorted(others)
    assert _alarms(rig) == []
