'''
Closed loop on DDS inside one process (no robot, no network): the real
ArmController against kinematic fake G1s. One fake's commanded joints (kp > 0)
jump to their targets, and it relays every command on rt/arm_sdk at weight 1
like an engaged safety layer: startup, the waist hold, a palm target, estop,
and a move home cut short by controller.startup_move_timeout. Without
rt/lowstate the controller commands nothing and stops with an error. With
nothing relaying its commands, they stay near the swaying fake, and the
start-up stops with an error at controller.startup_timeout, well before
controller.startup_move_timeout. The others sit behind the real
g1_safety_layer. In arm_sdk mode (sharing the waist and arms with Unitree
through the weight slot) the controller waits for rt/lowstate and, unless
follow_tolerance turns the gate off, never commands far from the measured pose
before and at engagement. In split mode (with a fake walking node on the legs)
the gate opens only once the walking node commands too, and when it opens 20 s
into the start-up, the move home still finishes, though the start-up outlasts
controller.startup_move_timeout. In both modes, once engaged, the controller
holds against a push instead of following it
'''

import sys
import time
import threading
from pathlib import Path

import numpy as np
import pytest

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from g1_description.layout import ARM_SDK_WEIGHT_SLOT
from g1_control.utility import joint_definition as jd
from g1_control.core import low_cmd_handler
from g1_control.core.robot_model import RobotModel
from g1_control.core.channel_interface import LowCmdPublisher
from g1_control.utility.named_config import NAMED_CONFIGS
from g1_control.utility.path_definition import URDF_PATH
from g1_control.utility.controller_config import load_controller_config
from g1_control.core.controller.arm_controller import ArmController
from g1_control.core.controller.upper_controller import UpperController

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
SAFETY_DIR = Path(__file__).resolve().parents[2] / 'g1_safety_layer'
SAFETY_CONFIG_DIR = SAFETY_DIR / 'config'
# the sibling g1_safety_layer source, when the workspace is not sourced
if SAFETY_DIR.is_dir() and str(SAFETY_DIR) not in sys.path:
    sys.path.append(str(SAFETY_DIR))
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
UPPER = jd.UPPER_BODY_INDEX
# DDS latency, and the fake's motion since the controller's sample (rad)
SLACK = 0.025
needs_urdf = pytest.mark.skipif(
    not Path(URDF_PATH).is_file(), reason='leprechaun_g1 not checked out'
)
_factory = []


def _init_dds():
    # the Unitree channel factory is process-wide; initialize it once
    if not _factory:
        ChannelFactoryInitialize(TEST_DOMAIN)
        _factory.append(True)


class FakeG1:
    '''
    Publish rt/lowstate; joints commanded with kp > 0 jump to their q (the
    waist sags). Stands in for an engaged g1_safety_layer too: the last
    command is relayed on rt/arm_sdk at weight 1
    '''

    def __init__(self, q0, waist_sag=(0.0, 0.0, 0.0)):
        self.q = np.asarray(q0, dtype=float).copy()
        self.waist_sag = np.asarray(waist_sag, dtype=float)
        self.lock = threading.Lock()
        self.commands = 0
        self.zero_gain_commands = 0
        self.last_cmd_time = None
        self.last_cmd = None
        self.stop = threading.Event()
        self.state_pub = ChannelPublisher('rt/lowstate', LowState_)
        self.state_pub.Init()
        self.relay_pub = ChannelPublisher('rt/arm_sdk', LowCmd_)
        self.relay_pub.Init()
        self.cmd_sub = ChannelSubscriber('rt/safety/lowcmd_upper_in', LowCmd_)
        self.cmd_sub.Init(self._on_cmd, 10)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _on_cmd(self, msg):
        with self.lock:
            self.commands += 1
            self.last_cmd_time = time.time()
            if max(msg.motor_cmd[i].kp for i in jd.UPPER_BODY_INDEX) == 0.0:
                self.zero_gain_commands += 1
            for i in range(jd.NUM_MOTOR):
                if msg.motor_cmd[i].kp > 0.0:
                    self.q[i] = msg.motor_cmd[i].q
            self.q[jd.WAIST_INDEX] -= self.waist_sag
            msg.motor_cmd[ARM_SDK_WEIGHT_SLOT].q = 1.0
            self.last_cmd = msg

    def _loop(self):
        state = LowStateDefault()
        state.mode_machine = 5
        state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        while not self.stop.is_set():
            with self.lock:
                for i in range(jd.NUM_MOTOR):
                    state.motor_state[i].q = float(self.q[i])
                relay = self.last_cmd
            self.state_pub.Write(state)
            if relay is not None:
                self.relay_pub.Write(relay)
            time.sleep(0.002)

    def arm_q(self):
        with self.lock:
            return self.q[jd.LEFT_ARM_INDEX[0]:].copy()

    def close(self):
        # close every channel: one left to the garbage collector keeps its DDS
        # entity, and the next channel on the same topic then segfaults
        self.stop.set()
        self.thread.join()
        self.state_pub.Close()
        self.relay_pub.Close()
        self.cmd_sub.Close()


class FakeRelayedG1:
    '''
    A G1 behind g1_safety_layer's output topic. Each waist and arm joint tracks
    weight x command + (1 - weight) x the hold, plus push (a force holding it
    off its target), with a TRACK_S lag. On rt/arm_sdk the weight is slot 29 of
    the last command, 0 until one arrives (the hold is Unitree's); on rt/lowcmd
    it is 1 from the first command on (Unitree's controller released, the hold
    is where the robot hangs). The hold sways sway rad on SWAY_SLOTS, so the
    measured pose drifts before the safety layer engages. rt/lowstate starts
    with start()
    '''

    TRACK_S = 0.05
    SWAY_PERIOD_S = 2.0
    SWAY_SLOTS = [jd.WAIST_INDEX[0], jd.LEFT_ARM_INDEX[1]]  # waist yaw, left shoulder roll

    def __init__(self, q0, topic='rt/arm_sdk', sway=0.1):
        self.q0 = np.asarray(q0, dtype=float).copy()
        self.q = self.q0.copy()
        self.cmd_q = self.q0.copy()
        self.push = np.zeros(jd.NUM_MOTOR)
        self.sway = sway
        self.weighted = topic == 'rt/arm_sdk'
        self.weight = 0.0
        self.lock = threading.Lock()
        self.started = None
        self.stop = threading.Event()
        self.state_pub = ChannelPublisher('rt/lowstate', LowState_)
        self.state_pub.Init()
        self.cmd_sub = ChannelSubscriber(topic, LowCmd_)
        self.cmd_sub.Init(self._on_cmd, 10)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.started = time.time()
        self.thread.start()

    def _on_cmd(self, msg):
        with self.lock:
            self.weight = float(msg.motor_cmd[ARM_SDK_WEIGHT_SLOT].q) if self.weighted else 1.0
            for i in UPPER:
                self.cmd_q[i] = msg.motor_cmd[i].q

    def _hold(self, t):
        hold = self.q0.copy()
        hold[self.SWAY_SLOTS] += self.sway * np.sin(2.0 * np.pi * t / self.SWAY_PERIOD_S)
        return hold

    def _loop(self):
        state = LowStateDefault()
        state.mode_machine = 5
        state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        last = time.time()
        while not self.stop.is_set():
            now = time.time()
            # the lag in wall time, however late this loop runs
            alpha = 1.0 - np.exp(-(now - last) / self.TRACK_S)
            last = now
            with self.lock:
                target = self.weight * self.cmd_q + (1.0 - self.weight) * self._hold(
                    now - self.started) + self.push
                self.q[UPPER] += (target[UPPER] - self.q[UPPER]) * alpha
                for i in range(jd.NUM_MOTOR):
                    state.motor_state[i].q = float(self.q[i])
            self.state_pub.Write(state)
            time.sleep(0.002)

    def set_push(self, push):
        with self.lock:
            self.push = np.asarray(push, dtype=float).copy()

    def measured(self):
        with self.lock:
            return self.q.copy()

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()
        self.state_pub.Close()
        self.cmd_sub.Close()


class FakeWalkingNode:
    '''Command the legs at their q0 on the safety layer's lower input from start()'''

    def __init__(self, q0):
        self.cmd = LowCmdDefault()
        for i in range(12):
            self.cmd.motor_cmd[i].mode = 1
            self.cmd.motor_cmd[i].q = float(q0[i])
            self.cmd.motor_cmd[i].kp = 100.0
            self.cmd.motor_cmd[i].kd = 2.0
        self.stop = threading.Event()
        self.pub = ChannelPublisher('rt/safety/lowcmd_lower_in', LowCmd_)
        self.pub.Init()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            self.pub.Write(self.cmd)
            time.sleep(0.005)

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()
        self.pub.Close()


class CommandRecorder:
    '''(time, upper-body q, weight slot, fake G1's measured upper-body q) per command on topic'''

    def __init__(self, topic, robot):
        self.robot = robot
        self.samples = []
        self.sub = ChannelSubscriber(topic, LowCmd_)
        self.sub.Init(self._on_cmd, 50)

    def _on_cmd(self, msg):
        q = np.array([msg.motor_cmd[i].q for i in UPPER])
        measured = self.robot.measured()[UPPER] if hasattr(self.robot, 'measured') else None
        weight = float(msg.motor_cmd[ARM_SDK_WEIGHT_SLOT].q)
        self.samples.append((time.time(), q, weight, measured))

    def close(self):
        self.sub.Close()


def _watch_gate_opening(monkeypatch):
    '''
    Return the list of (time, weight) of the safety layer output samples that
    open a follow gate: once open, the controller checks no more samples
    '''
    opened = []
    relays_commands = low_cmd_handler.relays_commands

    def relays_and_record(sample_q, sample_kp, weight, *args):
        relays = relays_commands(sample_q, sample_kp, weight, *args)
        if relays:
            opened.append((time.time(), weight))
        return relays
    monkeypatch.setattr(low_cmd_handler, 'relays_commands', relays_and_record)
    return opened


def _watch_subscription(monkeypatch):
    '''Return an event set once a controller subscribes to rt/lowstate'''
    subscribed = threading.Event()
    init_subscriber = RobotModel.init_subscriber

    def init_subscriber_and_tell(self, *args, **kwargs):
        init_subscriber(self, *args, **kwargs)
        subscribed.set()
    monkeypatch.setattr(RobotModel, 'init_subscriber', init_subscriber_and_tell)
    return subscribed


def _watch_startup(monkeypatch):
    '''
    Return a dict that gets the start-up routine's start and end times, with
    an event under 'started' set once it starts
    '''
    times = {'started': threading.Event()}
    startup_routine = UpperController._startup_routine

    def startup_and_time(self, *args, **kwargs):
        times['start'] = time.time()
        times['started'].set()
        try:
            startup_routine(self, *args, **kwargs)
        finally:
            times['end'] = time.time()
    monkeypatch.setattr(UpperController, '_startup_routine', startup_and_time)
    return times


def _build_in_thread(config):
    '''Start building an ArmController (init=True) in a thread; return it and the result dict'''
    built = {}

    def build():
        try:
            built['controller'] = ArmController(URDF_PATH, init=True, config=config)
        except BaseException as err:  # reported by the main thread
            built['error'] = err
    builder = threading.Thread(target=build, daemon=True)
    builder.start()
    return builder, built


def _shutdown_built(builder, built):
    '''Shut down a controller built by _build_in_thread, once its build is over'''
    if builder is not None and builder.is_alive():
        builder.join(45.0)
    if 'controller' in built:
        built['controller'].shutdown()


def _push_and_check_the_hold(controller, robot, recorders):
    '''
    Once engaged, push the waist roll and pitch and the left elbow well past
    the follow tolerance: every command stays where it was, and the joints
    move by the push and no further
    '''
    push = np.zeros(jd.NUM_MOTOR)
    push[jd.WAIST_INDEX[1]] = 0.2  # waist roll, 4 x the follow tolerance
    push[jd.WAIST_INDEX[2]] = -0.1  # waist pitch
    push[jd.LEFT_ARM_INDEX[3]] = 0.15  # left elbow
    held = controller.low_cmd_handler.q_cmd[UPPER].astype(np.float64)
    # the waist holds its target itself (as float32 on DDS)
    assert np.max(np.abs(held[:3] - controller.waist_target)) < 1e-6
    # every recorder has caught up with the held command
    deadline = time.time() + 2.0
    while time.time() < deadline and not all(
            np.array_equal(r.samples[-1][1], held) for r in recorders):
        time.sleep(0.01)
    start = time.time()
    robot.set_push(push)
    # the joints move by the push and no further, and stay there
    assert _settles(robot, held + push[UPPER])
    time.sleep(1.0)
    pushed = robot.measured()
    robot.set_push(np.zeros(jd.NUM_MOTOR))
    end = time.time()
    assert np.max(np.abs(pushed[UPPER] - (held + push[UPPER]))) < 0.01
    # and back to the hold once let go
    assert _settles(robot, held)
    for recorder in recorders:
        during = np.array([q for t, q, _, _ in recorder.samples if start <= t <= end])
        assert len(during) > 100
        assert np.max(np.abs(during - held)) < 1e-6, 'a held command followed the push'


def _settles(robot, upper_q, tolerance=0.01, timeout=5.0):
    '''True once the fake's upper-body q is within tolerance of upper_q'''
    deadline = time.time() + timeout
    while time.time() < deadline:
        if np.max(np.abs(robot.measured()[UPPER] - upper_q)) < tolerance:
            return True
        time.sleep(0.02)
    return False


@needs_urdf
def test_startup_home_palm_target_and_estop_silence():
    _init_dds()
    # start like a standing G1 under Unitree's control: arms hanging, waist turned a bit
    q0 = np.zeros(jd.NUM_MOTOR)
    q0[jd.LEFT_ARM_INDEX[0]:] = NAMED_CONFIGS['arms_down']
    q0[jd.WAIST_INDEX] = [0.05, 0.0, 0.0]
    robot = FakeG1(q0)
    time.sleep(0.3)

    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    controller = ArmController(URDF_PATH, init=True, config=config)
    try:
        # startup: arms at home, waist eased to its 0 hold, never a zero-gain command
        assert np.max(np.abs(robot.arm_q() - NAMED_CONFIGS['home'])) < 0.02
        assert np.max(np.abs(robot.q[jd.WAIST_INDEX])) < 1e-2
        assert robot.commands > 0 and robot.zero_gain_commands == 0
        # the commands were relayed on rt/arm_sdk at weight 1: the gate is open
        assert controller.low_cmd_handler.following

        # a left palm target 5 cm forward is reached; the right palm stays put
        controller.ik_solver.update_configurations()
        controller.ik_solver.set_from_configuration()
        right_start = controller.right_ee_position.copy()
        target = controller.left_ee_position + np.array([0.05, 0.0, 0.0])
        controller.left_ee_target_position = target
        for _ in range(150):
            controller.control_step_reduced()
            time.sleep(controller.dt)
        assert np.linalg.norm(controller.left_ee_position - target) < 0.005
        assert np.linalg.norm(controller.right_ee_position - right_start) < 0.005

        # estop: the controller goes silent instead of sending zero-gain commands
        controller.estop()
        time.sleep(0.1)
        silent_since = time.time()
        time.sleep(0.3)
        assert robot.last_cmd_time < silent_since
        assert robot.zero_gain_commands == 0
    finally:
        controller.shutdown()
        robot.close()


@needs_urdf
def test_no_lowstate_means_no_command_and_a_clear_error():
    _init_dds()
    recorder = CommandRecorder('rt/safety/lowcmd_upper_in', robot=None)
    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    config['controller']['state_timeout'] = 0.5
    try:
        with pytest.raises(RuntimeError, match='No fresh rt/lowstate .* within 0.5 s .*'
                                               'not enabling any motor'):
            ArmController(URDF_PATH, init=True, config=config)
        time.sleep(0.2)
        assert recorder.samples == [], 'nothing may be commanded without rt/lowstate'
    finally:
        recorder.close()


@needs_urdf
def test_startup_stops_with_an_error_when_the_safety_layer_never_engages(monkeypatch):
    _init_dds()
    # the publisher stops before rt/lowstate goes: the robot model's local
    # all-zero pose must never reach the follow gate while commands go out
    order = []
    for cls, name in ((LowCmdPublisher, 'publisher'), (RobotModel, 'state')):
        def shutdown_and_record(self, _shutdown=cls.shutdown, _name=name):
            order.append(_name)
            _shutdown(self)
        monkeypatch.setattr(cls, 'shutdown', shutdown_and_record)
    # nothing relays the commands to rt/arm_sdk: the gate stays shut, the
    # commands stay near the swaying robot, and the start-up gives up after its
    # own 2 s wait, not after the moves' 10 s budget
    q0 = np.zeros(jd.NUM_MOTOR)
    q0[jd.LEFT_ARM_INDEX[3]] = q0[jd.RIGHT_ARM_INDEX[3]] = 0.8  # elbows
    robot = FakeRelayedG1(q0)
    robot.start()
    recorder = CommandRecorder('rt/safety/lowcmd_upper_in', robot)
    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    config['controller']['startup_timeout'] = 2.0
    config['controller']['startup_move_timeout'] = 10.0
    startup = _watch_startup(monkeypatch)
    try:
        with pytest.raises(RuntimeError, match=r'g1_safety_layer did not engage within '
                                               r'controller.startup_timeout \(2 s\): no '
                                               'rt/arm_sdk sample at weight 1'):
            ArmController(URDF_PATH, init=True, config=config)
        stopped = time.time()
        time.sleep(0.3)
        assert 2.0 <= startup['end'] - startup['start'] < 3.0
        tolerance = config['controller']['follow_tolerance']
        assert len(recorder.samples) > 100
        assert max(np.max(np.abs(q - m)) for _, q, _, m in recorder.samples) <= tolerance + SLACK
        assert all(t < stopped + 0.05 for t, _, _, _ in recorder.samples), 'still publishing'
        assert order == ['publisher', 'state']
    finally:
        recorder.close()
        robot.close()


@needs_urdf
def test_startup_move_stops_with_an_error_after_its_own_timeout():
    _init_dds()
    # relayed at once, so the gate opens at the first command; from hanging
    # arms (elbows pi/2 from home) the move home takes at least 5.2 s at 0.3 rad/s
    q0 = np.zeros(jd.NUM_MOTOR)
    q0[jd.LEFT_ARM_INDEX[0]:] = NAMED_CONFIGS['arms_down']
    robot = FakeG1(q0)
    time.sleep(0.3)
    recorder = CommandRecorder('rt/safety/lowcmd_upper_in', robot=None)
    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    config['controller']['startup_move_timeout'] = 2.0
    controller = None
    try:
        with pytest.raises(RuntimeError, match=r'Moving home: not done within '
                                               r'controller.startup_move_timeout \(2 s\) of the '
                                               'follow gate opening'):
            # one that does not raise is shut down below, before the next test
            controller = ArmController(URDF_PATH, init=True, config=config)
        stopped = time.time()
        time.sleep(0.3)
        # stopped mid-move, and silent since
        elbow = robot.q[jd.LEFT_ARM_INDEX[3]]
        assert 0.1 < elbow < NAMED_CONFIGS['arms_down'][3] - 0.1, elbow
        assert len(recorder.samples) > 50
        assert all(t < stopped + 0.05 for t, _, _, _ in recorder.samples), 'still publishing'
    finally:
        if controller is not None:
            controller.shutdown()
        recorder.close()
        robot.close()


@needs_urdf
def test_every_command_holds_the_waist_target_while_the_waist_sags():
    _init_dds()
    # waist pitch sags 2 cm-rad below every command; each goal re-seeds the IK
    # from the measured joints, which must not drag the waist command down
    q0 = np.zeros(jd.NUM_MOTOR)
    robot = FakeG1(q0, waist_sag=(0.0, 0.0, 0.02))
    time.sleep(0.3)
    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    controller = ArmController(URDF_PATH, init=True, config=config)
    recorder = CommandRecorder('rt/safety/lowcmd_upper_in', robot=None)
    try:
        for goal in range(3):
            controller.update_ik_solver()
            controller.ik_solver.set_from_configuration()
            target = controller.left_ee_position + np.array([0.02, 0.0, 0.0])
            controller.left_ee_target_position = target
            for _ in range(30):
                controller.control_step_reduced()
                time.sleep(controller.dt)
        waist = np.array([q[:3] for _, q, _, _ in recorder.samples])
        assert len(waist) > 100
        assert np.max(np.abs(waist - config['controller']['waist_target'])) < 1e-6
        # the measured waist sags once, it does not drift goal after goal
        assert abs(robot.q[jd.WAIST_INDEX[2]] + 0.02) < 1e-3
    finally:
        recorder.close()
        controller.shutdown()
        robot.close()


@needs_urdf
@pytest.mark.parametrize('follow_tolerance', [0.05, 0.0], ids=['gate', 'no_gate'])
def test_arm_sdk_waits_for_state_and_tracks_the_robot_through_engagement(
        monkeypatch, follow_tolerance):
    safety = pytest.importorskip('g1_safety_layer.core.safety_layer')
    safety_config = pytest.importorskip('g1_safety_layer.core.config')
    _init_dds()
    # a G1 standing under Unitree: arms hanging forward, elbows bent, waist turned
    q0 = np.zeros(jd.NUM_MOTOR)
    q0[jd.LEFT_ARM_INDEX] = [0.3, 0.25, 0.0, 0.8, 0.0, 0.0, 0.0]
    q0[jd.RIGHT_ARM_INDEX] = [0.3, -0.25, 0.0, 0.8, 0.0, 0.0, 0.0]
    q0[jd.WAIST_INDEX] = [0.04, 0.0, 0.0]
    robot = FakeRelayedG1(q0)
    upper_in = CommandRecorder('rt/safety/lowcmd_upper_in', robot)
    arm_sdk = CommandRecorder('rt/arm_sdk', robot)
    layer = safety.SafetyLayer(_safety_config(safety_config, 'arm_sdk.yaml'))
    config = load_controller_config('g1.yaml', config_dir=CONFIG_DIR)
    config['controller']['follow_tolerance'] = follow_tolerance
    # know when the controller listens, so rt/lowstate starts only after that
    subscribed = _watch_subscription(monkeypatch)
    opened = _watch_gate_opening(monkeypatch)

    builder, built = None, {}
    layer_thread = threading.Thread(target=layer.start, daemon=True)
    try:
        builder, built = _build_in_thread(config)
        assert subscribed.wait(30.0)
        time.sleep(0.5)
        assert upper_in.samples == [], 'nothing may be commanded before rt/lowstate'
        robot.start()
        # nothing relays the commands yet: the robot does not follow for 2 s
        time.sleep(2.0)
        layer_started = time.time()
        layer_thread.start()
        # before the safety layer runs, from the controller's commands alone
        before = [(q, m) for t, q, _, m in upper_in.samples if t < layer_started]
        assert upper_in.samples[0][0] >= robot.started
        assert len(before) > 100
        gap_before = max(np.max(np.abs(q - m)) for q, m in before)
        if follow_tolerance == 0.0:
            # without the gate the start-up move runs open loop to home, far from
            # the robot that is not following: the jump the gate prevents (a
            # safety layer with an engage gate refuses it, one without jumps)
            assert gap_before > 0.3
            return

        builder.join(45.0)
        assert not builder.is_alive() and 'error' not in built, built.get('error')
        controller = built['controller']
        bound = follow_tolerance + SLACK
        # the measured pose swayed more than a command standing still could cover
        sway = [UPPER.index(i) for i in robot.SWAY_SLOTS]
        assert np.all(np.ptp(np.array([m[sway] for _, m in before]), axis=0) > 2 * bound)
        assert gap_before <= bound
        # through the weight ramp too, and with no jump at engagement
        full = next(t for t, _, weight, _ in arm_sdk.samples if weight >= 1.0)
        through = [(q, m) for t, q, _, m in upper_in.samples if t <= full]
        assert max(np.max(np.abs(q - m)) for q, m in through) <= bound
        _, q, _, m = arm_sdk.samples[0]
        assert np.max(np.abs(q - m)) <= bound, 'jump at engagement'

        # the gate opened on a sample of the safety layer's output at weight 1,
        # the end of the ramp
        assert controller.low_cmd_handler.following
        assert len(opened) == 1 and opened[0][1] >= 0.999
        # then the start-up move: arms home, waist at its target
        waist_target = config['controller']['waist_target']
        assert _settles(robot, np.concatenate([waist_target, NAMED_CONFIGS['home']]), 0.02)

        # engaged: the hold resists a push instead of following it
        _push_and_check_the_hold(controller, robot, [upper_in, arm_sdk])
    finally:
        _shutdown_built(builder, built)
        if layer_thread.is_alive():
            layer.stop()
        upper_in.close()
        arm_sdk.close()
        robot.close()


@needs_urdf
def test_split_mode_engages_with_the_walking_node_then_holds_against_a_push(monkeypatch):
    safety = pytest.importorskip('g1_safety_layer.core.safety_layer')
    safety_config = pytest.importorskip('g1_safety_layer.core.config')
    _init_dds()
    # hanging on the gantry already at home: the start-up has nothing to move,
    # and only the safety layer's rt/lowcmd can tell the controller it engaged
    q0 = np.zeros(jd.NUM_MOTOR)
    robot = FakeRelayedG1(q0, topic='rt/lowcmd', sway=0.0)
    robot.start()
    upper_in = CommandRecorder('rt/safety/lowcmd_upper_in', robot)
    lowcmd = CommandRecorder('rt/lowcmd', robot)
    layer = safety.SafetyLayer(_safety_config(safety_config, 'split.yaml'))
    legs = FakeWalkingNode(q0)
    config = load_controller_config('g1_split.yaml', config_dir=CONFIG_DIR)
    opened = _watch_gate_opening(monkeypatch)

    builder, built = None, {}
    layer_thread = threading.Thread(target=layer.start, daemon=True)
    try:
        layer_thread.start()
        builder, built = _build_in_thread(config)
        deadline = time.time() + 30.0
        while time.time() < deadline and len(upper_in.samples) < 100:
            time.sleep(0.05)
        # the upper input alone: split mode stays silent, the gate shut
        time.sleep(1.0)
        assert len(upper_in.samples) > 100 and lowcmd.samples == [] and opened == []
        assert builder.is_alive(), 'the start-up must wait for the safety layer'

        legs.start()
        builder.join(30.0)
        assert not builder.is_alive() and 'error' not in built, built.get('error')
        controller = built['controller']
        first = lowcmd.samples[0]
        # opened on rt/lowcmd (no weight), once the walking node commanded
        assert len(opened) == 1 and opened[0][1] is None
        assert controller.low_cmd_handler.following
        assert np.max(np.abs(first[1] - first[3])) <= config['controller']['follow_tolerance']

        # the reviewed failure: engaged at home, with no start-up motion, the
        # waist pushed past the tolerance must not drag its command along
        _push_and_check_the_hold(controller, robot, [upper_in, lowcmd])
    finally:
        _shutdown_built(builder, built)
        if layer_thread.is_alive():
            layer.stop()
        legs.close()
        upper_in.close()
        lowcmd.close()
        robot.close()


@needs_urdf
def test_split_mode_engaging_late_still_gets_the_whole_move_home(monkeypatch):
    safety = pytest.importorskip('g1_safety_layer.core.safety_layer')
    safety_config = pytest.importorskip('g1_safety_layer.core.config')
    _init_dds()
    # hanging on the gantry with the elbows 1.5 rad from home: the move home
    # takes at least 5 s at g1_split.yaml's 0.3 rad/s
    q0 = np.zeros(jd.NUM_MOTOR)
    q0[jd.LEFT_ARM_INDEX[3]] = q0[jd.RIGHT_ARM_INDEX[3]] = 1.5
    robot = FakeRelayedG1(q0, topic='rt/lowcmd', sway=0.0)
    robot.start()
    layer = safety.SafetyLayer(_safety_config(safety_config, 'split.yaml'))
    legs = FakeWalkingNode(q0)
    config = load_controller_config('g1_split.yaml', config_dir=CONFIG_DIR)
    # the walking node commands 20 s into a 23 s wait for the engagement; the
    # move home then ends over 24 s into the start-up, so a 24 s move budget
    # is enough only if it counts from the gate opening (the move took 5 to
    # 14 s in our runs, on a shared host)
    config['controller']['startup_timeout'] = 23.0
    config['controller']['startup_move_timeout'] = 24.0
    late = 20.0
    startup = _watch_startup(monkeypatch)
    opened = _watch_gate_opening(monkeypatch)

    builder, built = None, {}
    layer_thread = threading.Thread(target=layer.start, daemon=True)
    try:
        layer_thread.start()
        builder, built = _build_in_thread(config)
        assert startup['started'].wait(30.0)
        time.sleep(late)
        legs.start()
        builder.join(45.0)
        assert not builder.is_alive() and 'error' not in built, built.get('error')
        # engaged late, then moved home past the end of the wait for it and
        # past the move budget counted from the start of the start-up
        assert len(opened) == 1 and opened[0][0] - startup['start'] >= late
        assert startup['end'] - startup['start'] > config['controller']['startup_move_timeout']
        waist_target = config['controller']['waist_target']
        assert _settles(robot, np.concatenate([waist_target, NAMED_CONFIGS['home']]), 0.02)
        # the controller kept commanding: the safety layer never estopped
        assert not layer._estop, layer._estop_reason
    finally:
        _shutdown_built(builder, built)
        if layer_thread.is_alive():
            layer.stop()
        legs.close()
        robot.close()


def _safety_config(safety_config, name):
    '''One of the safety layer's real-robot configs, without its log files'''
    config = safety_config.load_config(SAFETY_CONFIG_DIR / name)
    config['logging']['enabled'] = False
    return config
