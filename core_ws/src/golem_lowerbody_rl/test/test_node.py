"""
Check lowerbody_controller_node in one process, against fake publishers on a test domain.

A helper node stands in for the rest of the robot: /lowstate from an upright H1-2 whose
legs stand near the policy's pre-pose and whose tick counts ms of the wall clock (the
node's clock here), golem_safety's /safety/heartbeat, the arm controller's
/safety/lowcmd_upper_in, frame_task_server's /left_ee_pose and the sim's band toggle,
served under a name of the test's own (TOGGLE_REMAP). The node runs ALMI, its runner
wrapped to record every step and reset, in a MultiThreadedExecutor as main() spins it.
Needs the package, golem_safety and torch installed.
"""

import os
import threading
import time

from geometry_msgs.msg import PoseStamped, Twist
import golem_lowerbody_rl.node as node_module
from golem_lowerbody_rl.node import LowerBodyControllerNode, make_executor
from golem_lowerbody_rl.policy import PolicyRunner
from golem_lowerbody_rl.profile import load_profile
from golem_lowerbody_rl.supervisor import Phase
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

# the shell's ROS_DOMAIN_ID, but never 0 (the real robot) or 1 (the sim): 97 then
_SHELL_DOMAIN = int(os.environ.get('ROS_DOMAIN_ID') or 0)
TEST_DOMAIN = _SHELL_DOMAIN if _SHELL_DOMAIN > 1 else 97
# other tests on the domain may serve /elastic_band/toggle too
TOGGLE_REMAP = '/elastic_band/toggle:=/test_lowerbody_node/toggle'
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
# s of wall time: the 2 s ramp, the settle and 1 s of steady /lowstate, with margin
START_TIMEOUT = 10.0
# s a held step waits at most for the test to let it go
HOLD_TIMEOUT = 1.0


class WatchedRunner(PolicyRunner):
    """Run the policy, record every step and reset, and count a reset that lands mid-step."""

    def __init__(self, *args, **kwargs):
        """Build the policy and empty records."""
        super().__init__(*args, **kwargs)
        self.inputs = []
        self.outputs = []
        self.resets = []  # each reset's seed command (None: re-seed from the last)
        self.overlaps = 0
        self._stepping = False
        # hold(): the next step sets held and waits inside the step until release is set
        self.held = threading.Event()
        self.release = threading.Event()
        self._hold = False

    def hold(self):
        """Make the next step wait, mid-step, until release is set."""
        self.held.clear()
        self.release.clear()
        self._hold = True

    def step(self, inp):
        """Step the policy; a held step waits inside the step first."""
        self._stepping = True
        try:
            if self._hold:
                self._hold = False
                self.held.set()
                self.release.wait(HOLD_TIMEOUT)
            out = super().step(inp)
        finally:
            self._stepping = False
        self.inputs.append(inp)
        self.outputs.append(out)
        return out

    def reset(self, seed_cmd=None):
        """Reset the policy; count it when a step is in progress."""
        if self._stepping:
            self.overlaps += 1
        self.resets.append(None if seed_cmd is None else np.asarray(seed_cmd).tolist())
        super().reset(seed_cmd)


class RecordingNode(LowerBodyControllerNode):
    """The node, its log lines kept in records."""

    def __init__(self, records, **kwargs):
        """Keep records, then build the node."""
        self.records = records
        super().__init__(**kwargs)

    def _log(self, level, message):
        """Record the line, then log it."""
        self.records.append((level, message))
        super()._log(level, message)


class HeldToggle:
    """Stand in for the node's toggle client: there at once, and the test answers the call."""

    def __init__(self):
        """Record no call yet."""
        self.calls = 0
        self.done_callback = None

    def service_is_ready(self):
        """Return that the toggle is there."""
        return True

    def call_async(self, _request):
        """Count the call; return a future whose done callback the test runs."""
        self.calls += 1
        return self

    def add_done_callback(self, callback):
        """Keep the node's reply callback."""
        self.done_callback = callback

    def result(self):
        """Return the toggle's answer: the band is off."""
        return Trigger.Response(success=True, message='disabled')


class Robot:
    """Stand in for the robot and the other nodes: publish what the node reads, record it."""

    def __init__(self, context, legs_q):
        """Create the helper node's publishers, subscriptions and timers."""
        self.node = rclpy.create_node('test_lowerbody_robot', context=context)
        profile = load_profile('h12')
        self.q = np.zeros(len(profile.body_joints))
        self.q[profile.slots(profile.legs)] = legs_q
        self.heartbeat = False
        self.upper = True
        self.ee_pose = True
        self.cmd = None
        self.lower = []  # (monotonic s, LowCmd) of /safety/lowcmd_lower_in
        self.toggle_calls = []
        self._t0 = time.monotonic()
        self._state = self.node.create_publisher(
            LowState, '/lowstate', QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self._beat = self.node.create_publisher(Bool, '/safety/heartbeat', 10)
        self._upper = self.node.create_publisher(LowCmd, '/safety/lowcmd_upper_in', 1)
        self._ee = self.node.create_publisher(PoseStamped, '/left_ee_pose', 10)
        self._cmd = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.node.create_subscription(LowCmd, '/safety/lowcmd_lower_in',
                                      lambda m: self.lower.append((time.monotonic(), m)), 10)
        self.node.create_timer(0.004, self._publish_state)
        self.node.create_timer(0.02, self._publish_inputs)
        self.node.create_timer(0.1, self._publish_beat)

    def _publish_state(self):
        """Publish /lowstate: upright, legs at q, the tick in ms of the wall clock."""
        msg = LowState()
        msg.tick = int((time.monotonic() - self._t0) * 1000.0) & 0xFFFFFFFF
        msg.mode_machine = 6
        for slot, value in enumerate(self.q):
            msg.motor_state[slot].q = float(value)
        msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        self._state.publish(msg)

    def _publish_inputs(self):
        """Publish the arm controller's command, /left_ee_pose and /cmd_vel as set."""
        if self.upper:
            self._upper.publish(LowCmd())
        if self.ee_pose:
            self._ee.publish(PoseStamped())
        if self.cmd is not None:
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.angular.z = self.cmd
            self._cmd.publish(twist)

    def _publish_beat(self):
        """Publish golem_safety's heartbeat when on."""
        if self.heartbeat:
            self._beat.publish(Bool(data=True))

    def serve_toggle(self, success, message):
        """Serve /elastic_band/toggle (remapped), answering success and message."""
        def serve(_request, response):
            self.toggle_calls.append(time.monotonic())
            response.success, response.message = success, message
            return response
        self.node.create_service(Trigger, '/elastic_band/toggle', serve)

    def latest_policy(self, timeout=2.0):
        """Return /lowerbody/active_policy as a subscriber that joins now gets it."""
        got = []
        sub = self.node.create_subscription(String, '/lowerbody/active_policy',
                                            lambda m: got.append(m.data), LATCHED)
        try:
            _wait(lambda: got, timeout)
        finally:
            self.node.destroy_subscription(sub)
        return got[-1] if got else None


def _wait(condition, timeout):
    """Poll condition until it holds or timeout s pass; return whether it held."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return bool(condition())


@pytest.fixture
def rig(monkeypatch):
    """
    Return a function that starts the node and the fake robot, both spinning.

    start(config, policy='almi') -> (node, robot); everything is torn down after the test.
    """
    monkeypatch.setattr(node_module, 'PolicyRunner', WatchedRunner)
    context = rclpy.Context()
    rclpy.init(args=['--ros-args', '-r', TOGGLE_REMAP], context=context,
               domain_id=TEST_DOMAIN)
    spinning = []

    def start(config, policy='almi'):
        """Start the node on config and policy, and the fake robot beside it."""
        node = RecordingNode([], context=context, parameter_overrides=[
            Parameter('robot', value='h12'), Parameter('config', value=config),
            Parameter('policy', value=policy)])
        # the legs stand 0.05 rad from the pre-pose: the ramp starts where they are
        robot = Robot(context, node.runner.prepose + 0.05)
        robot_executor = SingleThreadedExecutor(context=context)
        robot_executor.add_node(robot.node)
        for spun, executor in ((node, make_executor(node, context=context)),
                               (robot.node, robot_executor)):
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            spinning.append((spun, executor, thread))
        return node, robot

    yield start
    for spun, executor, thread in spinning:
        executor.shutdown()
        thread.join(timeout=5.0)
        spun.destroy_node()
    rclpy.shutdown(context=context)


def _running(node, robot):
    """Turn the heartbeat on and wait until the policy runs."""
    robot.heartbeat = True
    assert _wait(lambda: node.supervisor.phase is Phase.RUNNING, START_TIMEOUT), (
        node.supervisor.phase, node.supervisor.wait_reason, node.records[-5:])


def _legs(msg, slots):
    """Return a LowCmd's leg (q, kp, kd, mode, dq, tau) arrays in leg order."""
    cmds = [msg.motor_cmd[slot] for slot in slots]
    return tuple(np.array([getattr(c, field) for c in cmds])
                 for field in ('q', 'kp', 'kd', 'mode', 'dq', 'tau'))


def test_publishes_nothing_before_the_first_true_heartbeat(rig):
    """Silent without golem_safety's beat; the first command is the measured legs."""
    node, robot = rig('sim')
    loaded = [m for _, m in node.records if m.startswith('loaded run config ')]
    assert len(loaded) == 1 and loaded[0].endswith('policies/almi.yaml'), node.records
    assert 'config/h12/sim.yaml and policy ' in loaded[0]
    time.sleep(1.5)
    assert robot.lower == []
    assert node.supervisor.phase is Phase.WAITING
    robot.heartbeat = True
    assert _wait(lambda: robot.lower, 3.0)
    slots = load_profile('h12').slots(load_profile('h12').legs)
    q, *_ = _legs(robot.lower[0][1], slots)
    # ramp step 0 is where the legs are
    np.testing.assert_allclose(q, robot.q[slots], atol=1e-6)


def test_walks_on_cmd_vel_with_the_policy_targets(rig):
    """Running, the node hands the policy /cmd_vel and publishes exactly its targets."""
    node, robot = rig('sim')
    _running(node, robot)
    robot.cmd = (0.5, 0.0, 0.0)
    runner = node.runner
    assert _wait(lambda: any(inp.cmd[0] == 0.5 for inp in runner.inputs), 3.0)
    time.sleep(0.5)
    robot.cmd = None
    profile = load_profile('h12')
    slots = profile.slots(profile.legs)
    targets = {tuple(out.q.tolist()) for out in list(runner.outputs)}
    recent = [msg for _, msg in robot.lower[-10:]]
    assert len(recent) == 10
    for msg in recent:
        q, kp, kd, mode, dq, tau = _legs(msg, slots)
        assert tuple(q.astype(np.float32).tolist()) in targets
        np.testing.assert_allclose(kp, runner.kp, rtol=1e-6)
        np.testing.assert_allclose(kd, runner.kd, rtol=1e-6)
        assert np.all(mode == 1) and not np.any(dq) and not np.any(tau)
        assert (msg.mode_machine, msg.mode_pr) == (profile.mode_machine, profile.mode_pr)
    # the processed command is the policy's: ALMI's gain and smoothing move it off zero
    assert runner.command[0] > 0.0


def test_band_reset_is_serialised_with_steps(rig):
    """A band reply that lands mid-step resets the policy once, after that step."""
    node, robot = rig('sim')
    toggle = HeldToggle()
    node._band_client = toggle
    _running(node, robot)
    assert _wait(lambda: toggle.done_callback is not None, 3.0)
    runner = node.runner
    runner.hold()
    assert runner.held.wait(2.0)
    # the reply arrives on another thread while the step is in progress
    reply = threading.Thread(target=toggle.done_callback, args=(toggle,))
    reply.start()
    time.sleep(0.1)
    resets_mid_step = list(runner.resets)
    runner.release.set()
    reply.join(timeout=2.0)
    assert not reply.is_alive()
    time.sleep(0.2)
    # the engage reset (zero seed), then the band release's (re-seeded from the last command)
    assert resets_mid_step == [[0.0, 0.0, 0.0]]
    assert runner.resets == [[0.0, 0.0, 0.0], None]
    assert runner.overlaps == 0
    assert toggle.calls == 1
    released = [m for _, m in node.records if m.startswith('elastic band released')]
    assert released == ['elastic band released (/elastic_band/toggle: disabled): the policy '
                        'restarts from an empty history']
    assert node.supervisor.phase is Phase.RUNNING


def test_band_released_through_the_toggle_once_it_is_there(rig):
    """Due with no toggle service: one warning; once it is there, one call releases the band."""
    node, robot = rig('sim')
    _running(node, robot)
    assert _wait(lambda: any('not there yet' in m for _, m in node.records), 3.0)
    robot.serve_toggle(True, 'disabled')
    assert _wait(lambda: any(m.startswith('elastic band released') for _, m in node.records),
                 5.0), node.records[-5:]
    time.sleep(1.0)
    assert len(robot.toggle_calls) == 1
    assert node.runner.resets == [[0.0, 0.0, 0.0], None]
    assert sum('not there yet' in m for _, m in node.records) == 1
    assert node.supervisor.phase is Phase.RUNNING


@pytest.mark.parametrize('success, message, level, text', [
    (False, 'nope', 'error', 'elastic band release failed: the service answered "nope"'),
    (True, 'enabled', 'warn', '/elastic_band/toggle answered "enabled"'),
])
def test_band_reply_that_is_no_release_is_final(rig, success, message, level, text):
    """A failed toggle, or one that put the band on, is logged once and never called again."""
    node, robot = rig('sim')
    robot.serve_toggle(success, message)
    _running(node, robot)
    assert _wait(lambda: any(m.startswith(text) for _, m in node.records), 5.0)
    time.sleep(1.0)
    assert len(robot.toggle_calls) == 1
    assert [lvl for lvl, m in node.records if m.startswith(text)] == [level]
    # no band-release reset
    assert node.runner.resets == [[0.0, 0.0, 0.0]]


def test_active_policy_is_latched(rig):
    """A late subscriber gets idle before the policy runs and its name once it does."""
    node, robot = rig('sim')
    assert robot.latest_policy() == 'idle'
    _running(node, robot)
    assert _wait(lambda: robot.latest_policy(0.5) == 'almi', 3.0)


def test_confirm_service(rig):
    """The real run config starts the policy only on a confirm while holding and ready."""
    node, robot = rig('real')
    client = robot.node.create_client(Trigger, '/lowerbody/confirm_engage')
    assert client.wait_for_service(timeout_sec=5.0)

    def call():
        future = client.call_async(Trigger.Request())
        assert _wait(future.done, 5.0)
        return future.result()

    refused = call()
    assert not refused.success and refused.message.startswith('refused: the node is waiting')
    robot.heartbeat = True
    assert _wait(lambda: node.supervisor.wait_reason == 'the operator has not confirmed yet',
                 START_TIMEOUT), (node.supervisor.phase, node.supervisor.wait_reason)
    assert node.supervisor.phase is Phase.HOLDING
    assert robot.latest_policy() == 'idle'
    accepted = call()
    assert accepted.success, accepted.message
    assert node.supervisor.phase is Phase.RUNNING
    assert _wait(lambda: robot.latest_policy(0.5) == 'almi', 3.0)
    again = call()
    assert not again.success and again.message.startswith('refused: the node is running')
    assert any(m.startswith('/lowerbody/confirm_engage: confirmed') for _, m in node.records)


def test_imu_trim_is_live_and_bounded(rig):
    """The trim parameters start at the run config's values and refuse more than 15 deg."""
    node, _ = rig('real')
    trim = [node.get_parameter(f'imu_trim_{axis}_deg').value
            for axis in ('roll', 'pitch', 'yaw')]
    assert trim == [0.5, -2.5, 0.0]
    (result,) = node.set_parameters([Parameter('imu_trim_pitch_deg', value=20.0)])
    assert not result.successful
    (result,) = node.set_parameters([Parameter('imu_trim_pitch_deg', value=-3.0)])
    assert result.successful
    assert any('IMU trim' in m and '-3.000' in m for _, m in node.records)
