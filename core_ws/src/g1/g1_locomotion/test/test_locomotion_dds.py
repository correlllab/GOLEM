'''
The walking node's DDS side inside one process (no robot, no network): a fake
G1 on rt/lowstate, a fake arm controller on the upper input, and the real
g1_safety_layer in split mode (split.yaml), on DDS domain 97
'''

import itertools
import json
import math
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import yaml

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from g1_safety_layer.core.config import load_config as load_safety_config
from g1_safety_layer.core.safety_layer import SafetyLayer

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
SPLIT_CONFIG = Path(__file__).resolve().parents[2] / 'g1_safety_layer' / 'config' / 'split.yaml'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
# This one process runs the safety layer at 500 Hz, the fakes and the
# recorders, and the GIL delays rt/lowstate by tens of ms at times (up to
# 90 ms seen), which the real node, in a process of its own, does not see.
# So these tests relax g1.yaml's 50 ms gap and 25 ms latency, and the tick
# rate the start accepts (a late sample at either end of its 1 s window moves
# it by as much); test_locomotion.py checks the real values. A safety layer
# with an rt/lowstate watchdog (control.state_stale_estop_seconds, 0.1 s in
# split.yaml) gets MAX_STATE_GAP too, except where a test needs its own value
MAX_STATE_GAP = 0.25
MAX_LATENCY = 0.25
TICK_RATE = [0.5, 2.0]
# rt/lowstate samples per policy step when every one arrives
SAMPLES_PER_STEP = 10
LEGS = range(0, 12)
SHOULDER = 15  # left shoulder pitch, an arm slot
# slack for thread scheduling when timing a watchdog
LATE_S = 0.3
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


class Rig:
    '''
    What _rig() starts; log holds the node's (time, level, message), fed
    every sample the driver gave the controller: (tick, time, whether it
    returned targets), and latencies every command's (time, latency) as the
    driver measured it
    '''


@contextmanager
def _rig(legs, ramp_seconds=0.2, state_watchdog=MAX_STATE_GAP):
    '''
    The node's controller and driver (g1.yaml, with a short ramp and the
    relaxed timing above), the real SafetyLayer (split.yaml; its rt/lowstate
    watchdog, if it has one, at state_watchdog s, or split.yaml's for None), a
    fake G1 and a fake arm controller (not started), on topics of their own,
    so no channel of another test can match this run's
    '''
    _init_dds()
    suffix = f'/test_locomotion_dds/{next(_runs)}'
    safety = load_safety_config(SPLIT_CONFIG)
    safety['logging']['enabled'] = False
    safety['topics'] = {key: name + suffix for key, name in safety['topics'].items()}
    if state_watchdog is not None and 'state_stale_estop_seconds' in safety['control']:
        safety['control']['state_stale_estop_seconds'] = state_watchdog
    with open(CONFIG_DIR / 'g1.yaml') as f:
        raw = yaml.safe_load(f)
    raw['topics'] = {key: name + suffix for key, name in raw['topics'].items()}
    raw['startup'] = {'ramp_seconds': ramp_seconds, 'tick_rate': TICK_RATE}
    raw['faults'] = {'max_state_gap': MAX_STATE_GAP, 'max_latency': MAX_LATENCY}

    rig = Rig()
    rig.safety = safety
    rig.policy = load_policy(str(POLICY_DIR))
    rig.config = check_config(raw, rig.policy.config['cmd_range'])
    assert rig.config['topics']['low_cmd'] == safety['topics']['low_cmd_lower_in']
    assert rig.config['topics']['upper_cmd'] == safety['topics']['low_cmd_upper_in']
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
    rig.lowcmd = Recorder(safety['topics']['low_cmd_out'])
    rig.lower = Recorder(rig.config['topics']['low_cmd'])
    layer = SafetyLayer(safety)
    threading.Thread(target=layer.start, daemon=True).start()
    rig.robot = FakeG1(rig.config['topics']['low_state'], legs)
    rig.arms = FakeArms(rig.config['topics']['upper_cmd'])
    rig.driver = LocomotionDriver(controller, rig.config['topics'], TEST_DOMAIN)
    try:
        # the layer's publisher loop starts after its 0.2 s writer setup
        time.sleep(0.5)
        yield rig
    finally:
        rig.driver.shutdown()
        rig.arms.close()
        rig.robot.close()
        layer.stop()
        rig.lowcmd.close()
        rig.lower.close()


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
    '''Every motor off, the rt/lowcmd estop command'''
    return _is_off(msg.motor_cmd[:29])


def _estop_reasons(capsys):
    '''Why the safety layer estopped, from the JSON events it printed'''
    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in lines if line.startswith('{')]
    return [e['reason'] for e in events if e['event'] == 'estop_command_written']


def test_silent_until_upright_state_and_arms_then_ramp_and_walk(capsys):
    default = np.asarray(load_policy(str(POLICY_DIR)).default_angles, dtype=np.float64)
    legs = default + np.linspace(-0.2, 0.2, 12)
    with _rig(legs, ramp_seconds=0.2) as rig:
        kp, kd = rig.policy.gains(rig.policy.leg_joints)
        # upright rt/lowstate but no arm controller: the node is silent, so the
        # safety layer is too
        time.sleep(0.4)
        assert rig.lower.msgs == [] and rig.lowcmd.msgs == []
        # the arm controller commands, but the robot is tilted 40 deg
        rig.robot.set(quat=_tilted(40.0))
        rig.arms.start()
        time.sleep(0.4)
        assert rig.lower.msgs == [] and rig.lowcmd.msgs == []
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

    # the safety layer publishes from the node's first command (the measured
    # legs) on: the node's legs with the training gains and the arm
    # controller's arms, stamped with mode_pr 0 and rt/lowstate's mode_machine
    assert lowcmd
    np.testing.assert_allclose(_legs(lowcmd[0][1]), legs, atol=1e-6)
    sent = {tuple(np.round(_legs(m), 6)) for _, m in lower}
    for _, msg in lowcmd:
        assert (msg.mode_pr, msg.mode_machine) == (0, 5)
        assert tuple(np.round(_legs(msg), 6)) in sent
        assert all(msg.motor_cmd[i].mode == 1 and msg.motor_cmd[i].kp == pytest.approx(kp[i]) and
                   msg.motor_cmd[i].kd == pytest.approx(kd[i]) for i in LEGS)
        assert msg.motor_cmd[SHOULDER].kp == 60.0
    assert _estop_reasons(capsys) == []
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
def test_bad_state_silences_the_node_then_the_safety_layer_estops(inject, reason, capsys):
    default = load_policy(str(POLICY_DIR)).default_angles
    # a stale rt/lowstate meets split.yaml's own rt/lowstate watchdog, if the
    # safety layer has one; the other faults leave rt/lowstate flowing
    state_watchdog = None if inject is _stale else MAX_STATE_GAP
    with _rig(default, ramp_seconds=0.1, state_watchdog=state_watchdog) as rig:
        watchdog = rig.safety['control']['lower_stale_estop_seconds']
        state_watchdog = rig.safety['control'].get('state_stale_estop_seconds')
        rig.arms.start()
        _wait_for(lambda: rig.driver.phase == WALKING)
        time.sleep(0.3)
        inject_time = time.monotonic()
        inject(rig)
        time.sleep(MAX_STATE_GAP + watchdog + LATE_S)
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
    # the safety layer then zeroed every motor, and latched
    idx = next((i for i, (_, msg) in enumerate(lowcmd) if _is_zeroed(msg)), None)
    assert idx is not None and idx > 0, 'the safety layer must estop'
    assert all(_is_zeroed(msg) for _, msg in lowcmd[idx:])
    reasons = _estop_reasons(capsys)
    assert len(reasons) == 1, reasons
    if reasons[0].startswith('Lower-body command stale'):
        # its leg watchdog, once the node had gone silent
        assert watchdog - 0.02 < lowcmd[idx][0] - last_lower < watchdog + LATE_S
    else:
        # or, for a stale rt/lowstate only, its own rt/lowstate watchdog,
        # which in split.yaml fires before the leg watchdog would
        assert inject is _stale and state_watchdog is not None, reasons
        assert reasons[0].startswith('rt/lowstate stale'), reasons
        assert state_watchdog - 0.02 < lowcmd[idx][0] - inject_time < state_watchdog + LATE_S


def test_a_backlog_is_dropped_and_the_cadence_kept(capsys):
    default = load_policy(str(POLICY_DIR)).default_angles
    with _rig(default, ramp_seconds=0.1) as rig:
        rig.arms.start()
        _wait_for(lambda: rig.driver.phase == WALKING)
        time.sleep(0.3)
        # the worker is held up for 120 ms while about 60 samples arrive (the
        # safety layer's leg watchdog is 200 ms)
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
    assert _estop_reasons(capsys) == []


def test_the_latency_counts_the_time_a_sample_waited_for_the_node(capsys):
    default = load_policy(str(POLICY_DIR)).default_angles
    with _rig(default, ramp_seconds=0.1) as rig:
        rig.arms.start()
        _wait_for(lambda: rig.driver.phase == WALKING)
        time.sleep(0.3)
        # the worker is held up, and rt/lowstate stops 30 ms in for 150 ms:
        # when the worker goes on, the newest sample is the one written 90 ms
        # before (within the relaxed gap and rt/lowstate watchdog, and the
        # safety layer's 200 ms leg watchdog)
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
    assert _estop_reasons(capsys) == []
