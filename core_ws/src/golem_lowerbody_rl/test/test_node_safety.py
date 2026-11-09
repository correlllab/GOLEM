"""
Check the G1's leg node behind golem_safety's installed safety_node on a test domain.

In this process: lowerbody_controller_node (robot g1, the sim run config, HOMIE) and a
helper node standing in for the rest of the G1 (/lowstate from a G1 whose tick counts ms of
the wall clock, the arm controller's /safety/lowcmd_upper_in, frame_task_server's
/left_ee_pose and the sim's band toggle), which also records /lowcmd, the node's leg input
and /safety/heartbeat. Beside them, golem_safety's safety_node runs as a subprocess with
robot g1 on a copy of its sim_split.yaml (split mode, sim:=true, no hardware estop), so the
node's commands reach the motors only as golem_safety relays them. Every topic and service
of a run is moved under a prefix of its own (the context's remaps, and the copy's topics),
so no other run or test on the domain meets it. Needs golem_safety and onnxruntime
installed, and a network on which DDS discovery works (not --network none).
"""

from contextlib import contextmanager
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading
import time

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import PoseStamped
from golem_lowerbody_rl.node import LowerBodyControllerNode, make_executor
from golem_lowerbody_rl.policy import load_policy_spec
from golem_lowerbody_rl.profile import load_profile
from golem_lowerbody_rl.supervisor import Phase, UPPER_KIND
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState
import yaml

# the shell's ROS_DOMAIN_ID, but never 0 (the real robot) or 1 (the sim): 97 then
_SHELL_DOMAIN = int(os.environ.get('ROS_DOMAIN_ID') or 0)
TEST_DOMAIN = _SHELL_DOMAIN if _SHELL_DOMAIN > 1 else 97
# every name the node, the helper or safety_node uses, moved under a run's prefix
NAMES = ('/lowstate', '/lowcmd', '/safety/lowcmd_lower_in', '/safety/lowcmd_upper_in',
         '/safety/heartbeat', '/left_ee_pose', '/elastic_band/toggle', '/cmd_vel',
         '/lowerbody/height', '/lowerbody/active_policy', '/lowerbody/confirm_engage')
PROFILE = load_profile('g1')
LEG_SLOTS = [int(slot) for slot in PROFILE.slots(PROFILE.legs)]
# rad, leg order: HOMIE's default pose, the node's pre-pose (no target clip)
PREPOSE = np.array([load_policy_spec('g1', 'homie').default_pose[leg] for leg in PROFILE.legs])
SHOULDER = 15  # left shoulder pitch, an arm slot
# s of /lowstate period: 250 Hz, a tick of 4 ms a sample
STATE_PERIOD = 0.004
# s of wall time for the 2 s ramp, the settle and 1 s of steady /lowstate, with margin
START_TIMEOUT = 15.0
# slack for thread scheduling when timing a report of golem_safety's or the node's
LATE_S = 0.3
# the events by which golem_safety latches its estop or reports a quiet input or a
# quiet /lowstate: none may appear while the legs run
ALARMS = ('estop_command_written', 'input_quiet', 'state_quiet')
# rcutils' default console format: [LEVEL] [seconds] [logger]: message
LOG_LINE = re.compile(r'^\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[([0-9.]+)\] \[[^\]]*\]: (.*)$')
_runs = itertools.count()


def _tilted(degrees):
    """Return a pelvis quaternion (w, x, y, z) rolled by degrees about x."""
    half = math.radians(degrees) / 2
    return [math.cos(half), math.sin(half), 0.0, 0.0]


def _wait(condition, timeout):
    """Poll condition until it holds or timeout s pass; return whether it held."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return bool(condition())


class SafetyNode:
    """
    Run golem_safety's installed safety_node: robot g1 on the config at config, sim:=true.

    Its /safety/heartbeat is remapped to heartbeat; a thread collects its log lines.
    """

    def __init__(self, config, heartbeat):
        """Start the node."""
        executable = Path(get_package_prefix('golem_safety')) / 'lib' / 'golem_safety' / \
            'safety_node'
        env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN), RCUTILS_COLORIZED_OUTPUT='0')
        env.pop('RCUTILS_CONSOLE_OUTPUT_FORMAT', None)
        env.pop('RCUTILS_LOGGING_USE_STDOUT', None)
        self.lines = []
        self.exit_code = None
        self.proc = subprocess.Popen(
            [str(executable), '--ros-args', '-p', 'robot:=g1', '-p', f'config:={config}',
             '-p', 'sim:=true', '-p', 'use_estop:=false',
             '-r', f'/safety/heartbeat:={heartbeat}'],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        """Collect the node's output lines until it closes."""
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def output(self):
        """Return the node's output so far."""
        return '\n'.join(self.lines)

    def events(self, names=None):
        """Return the JSON events the node logged, in order; only those named in names."""
        events = []
        for line in list(self.lines):
            match = LOG_LINE.match(line)
            if match and match.group(3).startswith('{'):
                event = json.loads(match.group(3))
                if names is None or event['event'] in names:
                    events.append(event)
        return events

    def wait_for(self, name, timeout=15.0):
        """Wait until the node logs event name."""
        deadline = time.monotonic() + timeout
        while not self.events((name,)):
            assert self.proc.poll() is None, f'safety_node exited:\n{self.output()}'
            assert time.monotonic() < deadline, f'no {name} within {timeout} s:\n{self.output()}'
            time.sleep(0.02)

    def stop(self):
        """Stop the node with SIGINT, as Ctrl-C (SIGKILL if it hangs); keep its exit code."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._reader.join(timeout=5.0)
        self.exit_code = self.proc.returncode


class FakeG1:
    """
    Stand in for the G1, its arm controller and the sim; record what golem_safety writes.

    /lowstate every STATE_PERIOD from a G1 standing still (mode_machine 5, its tick ms of
    the wall clock), the arm controller's command (waist and arms at 0, kp 60) at 100 Hz
    while arms is set, /left_ee_pose while ee_pose is set, and the band toggle. Records
    /lowcmd, /safety/lowcmd_lower_in and /safety/heartbeat with their receive times.
    """

    def __init__(self, context, legs, quat):
        """Create the helper node's publishers, subscriptions, service and timers."""
        self.node = rclpy.create_node('test_node_safety_g1', context=context)
        self.lock = threading.Lock()
        self.q = np.zeros(len(PROFILE.body_joints))
        self.q[LEG_SLOTS] = legs
        self.quat = list(quat)
        self.gyro = [0.0, 0.0, 0.0]
        self.arms = False
        self.ee_pose = False
        self.lowcmd = []  # (monotonic s, LowCmd) of /lowcmd
        self.lower = []  # (monotonic s, LowCmd) of /safety/lowcmd_lower_in
        self.beats = []  # (monotonic s, bool) of /safety/heartbeat
        self.toggle_calls = []
        self._resume = 0.0
        self._t0 = time.monotonic()
        self._arm_cmd = LowCmd()
        for slot in range(12, len(PROFILE.body_joints)):
            self._arm_cmd.motor_cmd[slot].mode = 1
            self._arm_cmd.motor_cmd[slot].kp = 60.0
            self._arm_cmd.motor_cmd[slot].kd = 1.5
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._state = self.node.create_publisher(LowState, '/lowstate', newest)
        self._upper = self.node.create_publisher(LowCmd, '/safety/lowcmd_upper_in', 1)
        self._ee = self.node.create_publisher(PoseStamped, '/left_ee_pose', 10)
        self.node.create_subscription(
            LowCmd, '/lowcmd', lambda m: self.lowcmd.append((time.monotonic(), m)),
            QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.node.create_subscription(
            LowCmd, '/safety/lowcmd_lower_in', lambda m: self.lower.append((time.monotonic(), m)),
            10)
        self.node.create_subscription(
            Bool, '/safety/heartbeat', lambda m: self.beats.append((time.monotonic(), m.data)),
            10)
        self.node.create_service(Trigger, '/elastic_band/toggle', self._toggle)
        self.node.create_timer(STATE_PERIOD, self._publish_state)
        self.node.create_timer(0.01, self._publish_inputs)

    def set(self, quat=None, gyro=None, motor=None):
        """Change the IMU quaternion or gyro, or a motor's q: motor = (slot, q)."""
        with self.lock:
            if quat is not None:
                self.quat = list(quat)
            if gyro is not None:
                self.gyro = list(gyro)
            if motor is not None:
                self.q[motor[0]] = motor[1]

    def pause(self, seconds):
        """Publish no /lowstate for seconds."""
        with self.lock:
            self._resume = time.monotonic() + seconds

    def _publish_state(self):
        """Publish /lowstate unless paused."""
        now = time.monotonic()
        with self.lock:
            if now < self._resume:
                return
            msg = LowState()
            msg.tick = int((now - self._t0) * 1000.0) & 0xFFFFFFFF
            msg.mode_machine = PROFILE.mode_machine
            for slot, value in enumerate(self.q):
                msg.motor_state[slot].q = float(value)
            msg.imu_state.quaternion = [float(v) for v in self.quat]
            msg.imu_state.gyroscope = [float(v) for v in self.gyro]
        self._state.publish(msg)

    def _publish_inputs(self):
        """Publish the arm controller's command and /left_ee_pose while they are on."""
        if self.arms:
            self._upper.publish(self._arm_cmd)
        if self.ee_pose:
            self._ee.publish(PoseStamped())

    def _toggle(self, _request, response):
        """Answer the band toggle: the band is off."""
        self.toggle_calls.append(time.monotonic())
        response.success, response.message = True, 'disabled'
        return response


class TimedNode(LowerBodyControllerNode):
    """The leg node, its log lines kept in records with their monotonic times."""

    def __init__(self, records, **kwargs):
        """Keep records, then build the node."""
        self.records = records
        super().__init__(**kwargs)

    def _log(self, level, message):
        """Record the line, then log it."""
        self.records.append((time.monotonic(), level, message))
        super()._log(level, message)


class Rig:
    """Hold what _rig started: safety (SafetyNode), node (TimedNode) and robot (FakeG1)."""


@contextmanager
def _rig(legs=PREPOSE, quat=(1.0, 0.0, 0.0, 0.0), state_seconds=None):
    """
    Start safety_node, then the fake G1 and the leg node, each run's names under a prefix.

    safety_node runs a copy of sim_split.yaml with chunk logging off and its topics under
    the prefix; stale.state_seconds is set to state_seconds when given. The G1 starts with
    its legs at legs (rad, leg order) and its IMU at quat (w, x, y, z). safety_node is
    stopped first at the end, so that the rest of the teardown adds nothing to its log.
    """
    prefix = f'/test_node_safety/run{next(_runs)}'
    share = Path(get_package_share_directory('golem_safety'))
    safety = yaml.safe_load((share / 'config' / 'g1' / 'sim_split.yaml').read_text())
    safety['logging']['enabled'] = False
    safety['topics'] = {key: prefix + name for key, name in safety['topics'].items()}
    if state_seconds is not None:
        safety['stale']['state_seconds'] = state_seconds
    remaps = [arg for name in NAMES for arg in ('-r', f'{name}:={prefix}{name}')]
    rig = Rig()
    rig.safety_config = safety
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / 'g1' / 'sim_split.yaml'
        config.parent.mkdir()
        config.write_text(yaml.safe_dump(safety))
        rig.safety = SafetyNode(config, prefix + '/safety/heartbeat')
        context = rclpy.Context()
        spinning = []
        try:
            rig.safety.wait_for('relay_started')
            rclpy.init(args=['--ros-args', *remaps], context=context, domain_id=TEST_DOMAIN)
            rig.node = TimedNode([], context=context, parameter_overrides=[
                Parameter('robot', value='g1'), Parameter('config', value='sim'),
                Parameter('policy', value='homie')])
            rig.robot = FakeG1(context, legs, quat)
            robot_executor = SingleThreadedExecutor(context=context)
            robot_executor.add_node(rig.robot.node)
            for spun, executor in ((rig.node, make_executor(rig.node, context=context)),
                                   (rig.robot.node, robot_executor)):
                thread = threading.Thread(target=executor.spin, daemon=True)
                thread.start()
                spinning.append((spun, executor, thread))
            yield rig
        finally:
            rig.safety.stop()
            for spun, executor, thread in spinning:
                executor.shutdown()
                thread.join(timeout=5.0)
                spun.destroy_node()
            if context.ok():
                rclpy.shutdown(context=context)


def _legs(msg):
    """Return a LowCmd's leg targets, in leg order."""
    return np.array([msg.motor_cmd[slot].q for slot in LEG_SLOTS])


def _is_off(motors):
    """Return whether every motor is off: mode 0 and every value 0."""
    return all(m.mode == 0 and m.q == m.dq == m.tau == m.kp == m.kd == 0.0 for m in motors)


def _has_gains(msg, kp, kd):
    """Return whether every leg of msg is on at the policy's gains."""
    return all(msg.motor_cmd[slot].mode == 1 and msg.motor_cmd[slot].kp == pytest.approx(kp[i])
               and msg.motor_cmd[slot].kd == pytest.approx(kd[i])
               for i, slot in enumerate(LEG_SLOTS))


def _errors(rig):
    """Return the node's error lines: (monotonic s, message)."""
    return [(t, m) for t, level, m in list(rig.node.records) if level == 'error']


def _running(rig):
    """Have the arm controller and frame_task_server up; wait until the policy runs."""
    rig.robot.arms = True
    rig.robot.ee_pose = True
    assert _wait(lambda: rig.node.supervisor.phase is Phase.RUNNING, START_TIMEOUT), (
        rig.node.supervisor.phase, rig.node.supervisor.wait_reason, rig.node.records[-5:])


def test_g1_legs_start_behind_safety_node():
    """
    SAFETY: the legs move only once golem_safety relays, the robot is upright, and run later.

    golem_safety beats and streams zero gains; the node stays silent while the robot is
    tilted, pre-poses the legs from where they are once it is upright, holds the pre-pose
    until the arm controller commands and frame_task_server is up, then runs HOMIE and
    releases the band once. golem_safety engages each input on its own first command and
    relays the node's legs with the policy's gains, stamped for the G1.
    """
    # the legs stand up to 0.2 rad from the pre-pose, spread: the ramp starts where they are
    legs = PREPOSE + np.linspace(-0.2, 0.2, len(LEG_SLOTS))
    with _rig(legs, quat=_tilted(40.0)) as rig:
        np.testing.assert_allclose(rig.node.runner.prepose, PREPOSE, atol=1e-6)
        kp, kd = rig.node.runner.kp, rig.node.runner.kd
        # golem_safety beats once /lowstate says G1, but the robot is tilted: silent
        assert _wait(lambda: len(rig.robot.beats) >= 10, 5.0), rig.safety.output()
        time.sleep(1.0)
        assert rig.robot.lower == []
        assert rig.node.supervisor.phase is Phase.WAITING
        assert rig.node.supervisor.wait_reason.startswith('the robot is not upright')
        assert all(data for _, data in rig.robot.beats)
        assert rig.robot.lowcmd and all(_is_off(m.motor_cmd[:29]) for _, m in rig.robot.lowcmd)

        # upright: the ramp, then the pre-pose held while the arm controller is silent
        rig.robot.set(quat=[1.0, 0.0, 0.0, 0.0])
        assert _wait(lambda: rig.node.supervisor.phase is Phase.HOLDING, START_TIMEOUT), (
            rig.node.supervisor.phase, rig.node.records[-5:])
        time.sleep(1.5)
        assert rig.node.supervisor.phase is Phase.HOLDING
        assert rig.node.supervisor.wait_reason.startswith(UPPER_KIND)

        # the arm controller commands and frame_task_server is up: HOMIE runs, the band goes
        rig.robot.arms = True
        rig.robot.ee_pose = True
        assert _wait(lambda: rig.node.supervisor.phase is Phase.RUNNING, START_TIMEOUT), (
            rig.node.supervisor.wait_reason, rig.node.records[-5:])
        assert _wait(lambda: any(m.startswith('elastic band released')
                                 for _, _, m in list(rig.node.records)), 5.0)
        time.sleep(0.5)
        start, before = time.monotonic(), len(rig.robot.lower)
        time.sleep(1.0)
        rate = (len(rig.robot.lower) - before) / (time.monotonic() - start)
        lower = list(rig.robot.lower)
        lowcmd = list(rig.robot.lowcmd)
        records = list(rig.node.records)
        toggles = list(rig.robot.toggle_calls)
        assert rig.node.supervisor.fault is None, rig.node.supervisor.fault
    engaged = [e['source'] for e in rig.safety.events(('input_engaged',))]
    alarms = rig.safety.events(ALARMS)
    assert rig.safety.exit_code == 0, rig.safety.output()

    # the ramp starts at the measured legs and moves each one monotonically to the pre-pose
    # (2 s at 50 Hz: 100 steps, then the pre-pose)
    ramp = np.array([_legs(m) for _, m in lower[:101]])
    np.testing.assert_allclose(ramp[0], legs, atol=1e-6)
    toward = np.sign(PREPOSE - legs)
    assert np.all(np.diff(ramp, axis=0) * toward >= -1e-6), 'must move toward the pre-pose'
    np.testing.assert_allclose(ramp[-1], PREPOSE, atol=1e-6)
    # every leg command: the leg slots only, mode 1, dq = tau = 0, the policy's gains
    for _, msg in lower:
        assert _has_gains(msg, kp, kd)
        assert all(msg.motor_cmd[slot].dq == msg.motor_cmd[slot].tau == 0.0
                   for slot in LEG_SLOTS)
        assert _is_off(msg.motor_cmd[12:])
    # one command per 20 ms of tick while running: HOMIE's 50 Hz
    assert 40.0 < rate < 55.0, f'{rate:.1f} Hz'
    # the band was released once, because frame_task_server is up
    calls = [m for _, _, m in records if m.startswith('releasing the elastic band: ')]
    assert calls == ['releasing the elastic band: the policy runs and frame_task_server is up '
                     '(/left_ee_pose)']
    assert len(toggles) == 1

    # golem_safety engaged the node's input on its first command, the arm controller's
    # later on its own; the legs were off until then and carry the node's targets with the
    # policy's gains from then on, and the arms the controller's once it commands. Every
    # command is stamped with the G1 profile's mode_pr 0 and mode_machine 5
    assert engaged == ['split_mode_lower', 'split_mode_upper']
    first = next(i for i, (_, m) in enumerate(lowcmd) if m.motor_cmd[LEG_SLOTS[0]].kp > 0.0)
    assert all(_is_off([m.motor_cmd[slot] for slot in LEG_SLOTS]) for _, m in lowcmd[:first])
    np.testing.assert_allclose(_legs(lowcmd[first][1]), legs, atol=1e-6)
    sent = {tuple(np.round(_legs(m), 6)) for _, m in lower}
    for _, msg in lowcmd:
        assert (msg.mode_pr, msg.mode_machine) == (0, PROFILE.mode_machine)
    for _, msg in lowcmd[first:]:
        assert tuple(np.round(_legs(msg), 6)) in sent
        assert _has_gains(msg, kp, kd)
    assert lowcmd[-1][1].motor_cmd[SHOULDER].kp == 60.0
    assert alarms == []
    assert [m for _, level, m in records if level == 'error'] == []


def _stop_safety(rig):
    """Stop safety_node, as Ctrl-C would."""
    rig.safety.stop()


def _latch_safety(rig):
    """Put the left hip pitch past sim_split.yaml's estop limit: golem_safety latches."""
    # 3.5 rad, beyond its URDF range (to 2.88 rad) and the 0.1 rad sim_split.yaml adds
    rig.robot.set(motor=(LEG_SLOTS[0], 3.5))


@pytest.mark.parametrize('inject, reason', [
    (_stop_safety, 'no /safety/heartbeat for '),
    (_latch_safety, "golem_safety's /safety/heartbeat is false: its estop has latched"),
], ids=['safety stopped', 'estop latched'])
def test_heartbeat_loss_stops_the_legs(inject, reason):
    """SAFETY: a running node stops commanding the legs for good once golem_safety's beat stops."""
    with _rig() as rig:
        _running(rig)
        time.sleep(0.5)
        inject_time = time.monotonic()
        inject(rig)
        assert _wait(lambda: rig.node.supervisor.phase is Phase.FAULTED, 3.0)
        time.sleep(0.3)
        lower = list(rig.robot.lower)
        beats = list(rig.robot.beats)
        errors = _errors(rig)
        fault = rig.node.supervisor.fault
    assert fault.startswith(reason), fault
    estops = rig.safety.events(('estop_command_written',))
    assert rig.safety.exit_code == 0, rig.safety.output()

    # the node latched once and sent nothing after that
    assert len(errors) == 1 and errors[0][1].startswith(f'fault: {reason}'), errors
    fault_time = errors[0][0]
    assert lower[-1][0] < fault_time + 0.05, 'nothing may be sent after the fault'
    if inject is _stop_safety:
        # heartbeat.timeout (0.5 s) after the last beat
        last_beat = beats[-1][0]
        assert last_beat < fault_time
        assert 0.5 <= fault_time - last_beat < 0.5 + LATE_S, (last_beat, fault_time)
        assert estops == []
    else:
        # golem_safety latched and wrote its estop command; its next beat, false, stopped
        # the node
        assert estops
        first_false = next(t for t, data in beats if not data)
        assert inject_time < first_false <= fault_time < first_false + LATE_S


def _nan_gyro(rig):
    """Make the IMU's gyro not finite."""
    rig.robot.set(gyro=[math.nan, 0.0, 0.0])


def _fall(rig):
    """Tilt the robot 70 deg."""
    rig.robot.set(quat=_tilted(70.0))


def _stale(rig):
    """Publish no /lowstate for longer than the run config's max_state_gap, then resume."""
    rig.robot.pause(rig.node._run.faults.max_state_gap + 0.15)


@pytest.mark.parametrize('inject, reason', [
    (_stale, '/lowstate stale'),
    (_nan_gyro, '/lowstate gyro is not finite'),
    (_fall, 'fell: gravity z'),
], ids=['stale', 'not finite', 'fallen'])
def test_bad_state_stops_the_legs_and_golem_safety_holds_their_last_command(inject, reason):
    """
    SAFETY: a bad /lowstate stops the node for good; golem_safety holds its last command.

    golem_safety does not estop on a quiet input: it keeps relaying the node's last leg
    command and reports the quiet input once, stale.lower_seconds after it.
    """
    # a stale /lowstate meets a quiet-/lowstate time shorter than the pause; the other
    # faults leave /lowstate flowing
    state_seconds = 0.25 if inject is _stale else None
    with _rig(state_seconds=state_seconds) as rig:
        lower_quiet = rig.safety_config['stale']['lower_seconds']
        state_quiet = rig.safety_config['stale']['state_seconds']
        max_gap = rig.node._run.faults.max_state_gap
        kp, kd = rig.node.runner.kp, rig.node.runner.kd
        _running(rig)
        time.sleep(0.5)
        inject_time = time.monotonic()
        inject(rig)
        time.sleep(max_gap + lower_quiet + LATE_S + 0.2)
        end = time.monotonic()
        lower = list(rig.robot.lower)
        lowcmd = list(rig.robot.lowcmd)
        errors = _errors(rig)
        fault = rig.node.supervisor.fault
        assert rig.node.supervisor.phase is Phase.FAULTED and fault.startswith(reason), fault
    events = rig.safety.events()
    assert rig.safety.exit_code == 0, rig.safety.output()

    # the node latched once, soon after the fault, and sent nothing after that
    assert len(errors) == 1 and errors[0][1].startswith(f'fault: {reason}'), errors
    fault_time = errors[0][0]
    assert inject_time < fault_time < inject_time + max_gap + 0.15
    last_lower = lower[-1][0]
    assert last_lower < fault_time + 0.05, 'nothing may be sent after the fault'
    # never a zero-gain or damping command from the node itself
    assert all(_has_gains(msg, kp, kd) for _, msg in lower)

    # golem_safety went on relaying the node's last leg command, with the policy's gains,
    # to the end, and never its estop command
    held = [(t, msg) for t, msg in lowcmd if t > last_lower + 0.05]
    assert held, 'golem_safety must keep relaying'
    last_legs = _legs(lower[-1][1])
    for _, msg in held:
        np.testing.assert_allclose(_legs(msg), last_legs, atol=1e-6)
        assert _has_gains(msg, kp, kd)
    assert held[0][0] < last_lower + 0.15 and held[-1][0] > end - 0.1, (
        last_lower, held[0][0], held[-1][0], end)
    rate = (len(held) - 1) / (held[-1][0] - held[0][0])
    assert rate > 250.0, f'{rate:.0f} Hz of held commands; golem_safety relays at 500 Hz'

    # and reported the quiet leg input once, lower_quiet s after the node's last command,
    # with no estop and nothing to resume it
    quiet = [e for e in events if e['event'] == 'input_quiet']
    assert [(e['source'], e['topic'], e['seconds']) for e in quiet] == [
        ('split_mode_lower', rig.safety_config['topics']['low_cmd_lower_in'], lower_quiet)], quiet
    assert lower_quiet <= quiet[0]['age'] < lower_quiet + LATE_S, quiet
    assert [e for e in events if e['event'] == 'input_resumed'] == []
    assert [e for e in events if e['event'].startswith('estop')] == []
    # a stale /lowstate is reported too, and its return; the relay never waited for it
    states = [(e['event'], e['topic']) for e in events
              if e['event'] in ('state_quiet', 'state_resumed')]
    if inject is _stale:
        low_state = rig.safety_config['topics']['low_state']
        assert states == [('state_quiet', low_state), ('state_resumed', low_state)], states
        (reported,) = [e for e in events if e['event'] == 'state_quiet']
        assert reported['seconds'] == state_quiet, reported
        assert state_quiet <= reported['age'] < state_quiet + LATE_S, reported
    else:
        assert states == [], states
