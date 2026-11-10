"""Give the tests their ROS side, on domain 92, and the helpers the controller tests share.

Ros is one rclpy context per test with an executor spinning its nodes in a thread;
FakeRobot publishes /lowstate; Recorder records a command topic; Echo plays golem_safety
relaying the controller's commands on its output topic. profile_of, shared_model (the
session's robot model, put back at rest after the test) and G1_LIMP_ARMS (where the G1's
limp arms hang) serve the controller tests.

For the controller's closed loop: FollowingRobot executes the controller's commands as an
engaged golem_safety would relay them, and relays them itself; RelayedRobot sits behind
golem_safety's output topic, with golem_safety's safety_node run as a subprocess
(SafetyNode); CommandRecorder also notes the robot's measured pose; send_goal is an action
client.

SimClock publishes /clock as a sim would, at a chosen rate of sim seconds per wall second, and
Ros.node(name, sim_time=True) makes a node that runs on it (use_sim_time).

Every node lives in the test's context, so they share one DDS participant and discover each
other at once.
"""
import itertools
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from unitree_hg.msg import LowCmd, LowState
import yaml

TEST_DOMAIN = 92
# keeps every sample: the recorder must see each command the controller sends
RECORD_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST, depth=10000)
# a writer as the Unitree SDK's and golem_safety's: reliable, keep-last 1
WRITER_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
_names = itertools.count()
# The G1's limp arms, where they settle with every motor off (MuJoCo, the with-hand MJCF, the
# pelvis held up, the legs at the sim's stance): straight down, the wrists resting on the
# hips, clear of the self-collision spheres (unlike arms_down). The IK joints' order, rad
G1_LIMP_ARMS = (-0.053, -0.0083, 0.052, 1.2508, -0.0411, -0.0995, 0.071,
                -0.0523, 0.0075, -0.0589, 1.2543, 0.037, -0.1056, -0.0747)


def profile_of(robot):
    """Return robot's RobotProfile."""
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def shared_model(ros, models, robot):
    """Return the session's model of robot (models fixture), put back at rest at the end."""
    model = models(robot)

    def rest():
        """Detach the state source and re-lock the reduced model at q = 0."""
        model.state_subscriber = None
        model.rebuild_reduced_model(np.zeros(model.num_motor))
    ros.on_close(rest)
    return model


def wait_for(condition, timeout, period=0.005):
    """Return True as soon as condition() is, False once timeout s have passed without it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(period)
    return bool(condition())


class Ros:
    """Hold one test's rclpy context on TEST_DOMAIN, its nodes and a spinning executor."""

    def __init__(self):
        """Init the context and start spinning the executor in a thread."""
        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=TEST_DOMAIN)
        # a goal holds one thread for as long as it runs
        self.executor = MultiThreadedExecutor(num_threads=8, context=self.context)
        self._nodes = []
        self._closers = []
        self.spin_error = None
        self._spinner = threading.Thread(target=self._spin, daemon=True)
        self._spinner.start()

    def _spin(self):
        """Spin the executor; an error it raises fails the test at close()."""
        try:
            self.executor.spin()
        except Exception as error:  # close() fails the test with it
            self.spin_error = error

    def node(self, name, sim_time=False):
        """Return a new spinning node named name (made unique), destroyed at the end.

        sim_time: run the node with use_sim_time, on /clock (SimClock).
        """
        overrides = [Parameter('use_sim_time', value=True)] if sim_time else []
        node = rclpy.create_node(f'{name}_{next(_names)}', context=self.context,
                                 parameter_overrides=overrides)
        return self.add(node)

    def add(self, node):
        """Spin node, created in this context, and destroy it when the test ends."""
        self._nodes.append(node)
        self.executor.add_node(node)
        return node

    def on_close(self, close):
        """Run close() when the test ends, before the nodes go; the last registered first."""
        self._closers.append(close)

    def close(self):
        """Run the closers, stop the executor, destroy the nodes and shut the context down.

        Raises the first closer's error, or AssertionError if the executor stopped spinning.
        """
        errors = []
        for close in reversed(self._closers):
            try:
                close()
            except Exception as error:  # every closer runs; the first error is raised
                errors.append(error)
        self.executor.shutdown(timeout_sec=2.0)
        self._spinner.join(timeout=2.0)
        for node in self._nodes:
            node.destroy_node()
        rclpy.shutdown(context=self.context)
        if self.spin_error is not None:
            raise AssertionError(f'the executor stopped spinning: {self.spin_error!r}')
        if errors:
            raise errors[0]


class SimClock:
    """Publish /clock as a sim does: one STEP_S step of sim time per message, from start().

    rate is sim seconds per wall second (2.0 runs the sim twice as fast as the wall clock);
    pause() stops /clock, resume() starts it again, step_back() sets it back by seconds.
    """

    STEP_S = 0.002

    def __init__(self, ros, rate=1.0, start_s=100.0):
        """Create the publisher; /clock starts at start_s s of sim time on start()."""
        self.node = ros.node('sim_clock')
        self._publisher = self.node.create_publisher(Clock, '/clock', 10)
        self._lock = threading.Lock()
        self._rate = float(rate)
        self._t_ns = int(start_s * 1e9)
        self._running = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        ros.on_close(self.stop)

    @property
    def now(self):
        """Return the last sim time published, s."""
        with self._lock:
            return self._t_ns * 1e-9

    def start(self):
        """Start publishing."""
        self._running.set()
        self._thread.start()

    def _run(self):
        """Publish a step every STEP_S / rate s of wall time on absolute deadlines."""
        next_at = time.monotonic()
        msg = Clock()
        while not self._stop.is_set():
            if self._running.is_set():
                with self._lock:
                    self._t_ns += int(self.STEP_S * 1e9)
                    t_ns = self._t_ns
                msg.clock.sec, msg.clock.nanosec = divmod(t_ns, 10**9)
                self._publisher.publish(msg)
            next_at += self.STEP_S / self._rate
            now = time.monotonic()
            if now - next_at > 0.05:
                next_at = now
            time.sleep(max(0.0, next_at - now))

    def pause(self):
        """Stop publishing /clock."""
        self._running.clear()

    def resume(self):
        """Publish /clock again."""
        self._running.set()

    def step_back(self, seconds):
        """Set sim time back by seconds."""
        with self._lock:
            self._t_ns -= int(seconds * 1e9)

    def stop(self):
        """Stop the thread."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class FakeRobot:
    """Publish /lowstate for a profile, on demand (publish) or at a fixed rate (start).

    Each message carries q, dq and tick, the profile's mode_machine and a level IMU.
    """

    def __init__(self, ros, profile, topic='/lowstate'):
        """Create the publisher on topic (reliable, depth 100) with q and dq at 0."""
        self.profile = profile
        self.node = ros.node('fake_robot')
        self._publisher = self.node.create_publisher(
            LowState, topic,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=100))
        count = len(profile.body_joints)
        self.q = np.zeros(count)
        self.dq = np.zeros(count)
        self._start = time.monotonic()
        self._stop = threading.Event()
        self._thread = None
        ros.on_close(self.stop)

    def tick_now(self):
        """Return the robot's millisecond tick: ms since this fake started, from 1000."""
        return 1000 + int((time.monotonic() - self._start) * 1000.0)

    def publish(self, q=None, dq=None, tick=None):
        """Publish one /lowstate: q and dq (default the current ones) and tick (default now)."""
        q = self.q if q is None else np.asarray(q, dtype=float)
        dq = self.dq if dq is None else np.asarray(dq, dtype=float)
        msg = LowState()
        msg.mode_pr = self.profile.mode_pr
        msg.mode_machine = self.profile.mode_machine
        msg.tick = self.tick_now() if tick is None else int(tick)
        msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        for slot in range(len(q)):
            motor = msg.motor_state[slot]
            motor.mode = 1
            motor.q = float(q[slot])
            motor.dq = float(dq[slot])
            motor.tau_est = 0.5 * float(slot)
        self._publisher.publish(msg)

    def start(self, rate_hz=200.0):
        """Publish the current q and dq at rate_hz until stop()."""
        period = 1.0 / rate_hz

        def run():
            """Publish on absolute deadlines until stopped."""
            next_at = time.monotonic()
            while not self._stop.is_set():
                self.publish()
                next_at += period
                time.sleep(max(0.0, next_at - time.monotonic()))
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop publishing at a rate."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class Recorder:
    """Record every LowCmd on a topic, with its receive time on time.monotonic()."""

    def __init__(self, ros, topic):
        """Subscribe to topic, keeping every sample."""
        self.node = ros.node('recorder')
        self._lock = threading.Lock()
        self._samples = []
        self.node.create_subscription(LowCmd, topic, self._on_cmd, RECORD_QOS)

    def _on_cmd(self, msg):
        """Record msg with its receive time."""
        with self._lock:
            self._samples.append((time.monotonic(), msg))

    @property
    def samples(self):
        """Return the (receive time, LowCmd) samples so far."""
        with self._lock:
            return list(self._samples)

    @property
    def count(self):
        """Return how many commands arrived."""
        with self._lock:
            return len(self._samples)

    @property
    def last(self):
        """Return the newest LowCmd, or None."""
        with self._lock:
            return self._samples[-1][1] if self._samples else None

    def since(self, start):
        """Return the messages received at or after monotonic time start."""
        return [msg for t, msg in self.samples if t >= start]


class Echo:
    """Play golem_safety relaying the controller: republish every in_topic command on out_topic.

    clip, a (low, high) pair of per-slot arrays, clips each slot's q as golem_safety does;
    zero_kp sends kp 0 on every slot, as golem_safety's zero stream does before it engages.
    """

    def __init__(self, ros, in_topic, out_topic, clip=None, zero_kp=False):
        """Subscribe to in_topic and publish on out_topic."""
        self.node = ros.node('echo')
        self.clip = clip
        self.zero_kp = zero_kp
        self._publisher = self.node.create_publisher(LowCmd, out_topic, WRITER_QOS)
        self.node.create_subscription(LowCmd, in_topic, self._relay, RECORD_QOS)

    def _relay(self, msg):
        """Republish msg, clipped and with kp zeroed as configured."""
        if self.clip is not None:
            low, high = self.clip
            for slot in range(len(low)):
                motor = msg.motor_cmd[slot]
                motor.q = float(min(max(motor.q, low[slot]), high[slot]))
        if self.zero_kp:
            for motor in msg.motor_cmd:
                motor.kp = 0.0
        self._publisher.publish(msg)


def upper_q(msg, profile):
    """Return the upper-slot q of a LowCmd, as float64."""
    return np.array([msg.motor_cmd[slot].q for slot in profile.upper_slots], dtype=np.float64)


def slots_of(profile, joints):
    """Return the body slots of joints."""
    return [profile.body_joints.index(joint) for joint in joints]


def body_pose(profile, arms=None, held=None):
    """Return a body q: 0, the IK joints at arms and the held joints at held.

    arms is a named configuration's name or one value per IK joint; held one value per held
    joint.
    """
    q = np.zeros(len(profile.body_joints))
    if arms is not None:
        values = profile.named_configs[arms] if isinstance(arms, str) else arms
        q[slots_of(profile, profile.enabled_joints)] = values
    if held is not None:
        q[slots_of(profile, profile.held_joints)] = held
    return q


def _lowstate(profile):
    """Return a LowState of profile's robot with every motor on and a level IMU."""
    msg = LowState()
    msg.mode_pr = profile.mode_pr
    msg.mode_machine = profile.mode_machine
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    for slot in range(len(profile.body_joints)):
        msg.motor_state[slot].mode = 1
    return msg


class _Loop:
    """Run step() at rate_hz from a thread until stop()."""

    def __init__(self, rate_hz, step):
        """Set up the thread; start() starts it."""
        self._period = 1.0 / rate_hz
        self._step = step
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        """Start the thread."""
        self._thread.start()

    def _run(self):
        """Run step() on absolute deadlines until stopped."""
        next_at = time.monotonic()
        while not self._stop.is_set():
            self._step()
            next_at += self._period
            time.sleep(max(0.0, next_at - time.monotonic()))

    def stop(self):
        """Stop the thread."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class FollowingRobot:
    """Play a robot behind an engaged golem_safety, and golem_safety's relay too.

    Every command on command_topic is executed at once: each body slot with kp > 0 jumps to
    its q, less its sag (rad, per slot). The newest command is relayed on relay_topic.
    /lowstate goes out at rate_hz, its tick in ms on the monotonic clock, until stop_state(),
    with the IMU orientation imu_quaternion (w, x, y, z; default identity).
    """

    def __init__(self, ros, profile, q0, relay_topic, command_topic='/safety/lowcmd_upper_in',
                 sag=None, rate_hz=500.0, imu_quaternion=None):
        """Start at body q0, publishing /lowstate at once."""
        self.profile = profile
        self.node = ros.node('following_robot')
        count = len(profile.body_joints)
        self._lock = threading.Lock()
        self._q = np.array(q0, dtype=np.float64)
        self._sag = np.zeros(count) if sag is None else np.array(sag, dtype=np.float64)
        self._upper = list(profile.upper_slots)
        self.commands = 0
        self.zero_gain_commands = 0  # every upper slot at kp 0
        self.last_command = None  # (receive time on time.monotonic(), LowCmd)
        self._relay = None
        self._state_on = True
        self._state = _lowstate(profile)
        if imu_quaternion is not None:
            self._state.imu_state.quaternion = [float(value) for value in imu_quaternion]
        self._start = time.monotonic()
        self._state_publisher = self.node.create_publisher(LowState, '/lowstate', WRITER_QOS)
        self._relay_publisher = self.node.create_publisher(LowCmd, relay_topic, WRITER_QOS)
        self.node.create_subscription(LowCmd, command_topic, self._on_command, RECORD_QOS)
        self._loop = _Loop(rate_hz, self._step)
        self._loop.start()
        ros.on_close(self._loop.stop)

    def _on_command(self, msg):
        """Execute and count a command, and keep it to relay."""
        now = time.monotonic()
        motors = msg.motor_cmd
        with self._lock:
            self.commands += 1
            if all(motors[slot].kp == 0.0 for slot in self._upper):
                self.zero_gain_commands += 1
            for slot in range(len(self._q)):
                if motors[slot].kp > 0.0:
                    self._q[slot] = motors[slot].q - self._sag[slot]
            self._relay = msg
            self.last_command = (now, msg)

    def _step(self):
        """Publish /lowstate (unless stopped) and relay the newest command."""
        with self._lock:
            q = self._q.copy()
            relay = self._relay
            state_on = self._state_on
        if state_on:
            self._state.tick = int((time.monotonic() - self._start) * 1000.0)
            for slot, value in enumerate(q):
                self._state.motor_state[slot].q = float(value)
            self._state_publisher.publish(self._state)
        if relay is not None:
            self._relay_publisher.publish(relay)

    def measured(self):
        """Return the robot's body q."""
        with self._lock:
            return self._q.copy()

    def set_sag(self, sag):
        """Set how far below its command each slot settles from the next command on, rad."""
        with self._lock:
            self._sag = np.asarray(sag, dtype=np.float64).copy()

    def stop_state(self):
        """Publish no more /lowstate; commands are still executed and relayed."""
        with self._lock:
            self._state_on = False


class RelayedRobot:
    """Play a robot behind golem_safety's output topic (/lowcmd).

    Each upper-body slot the newest command gives kp > 0 tracks its command; any other upper
    slot tracks the hold, where the robot hangs, limp. push is added to every upper slot's
    target, and each tracks its target with a TRACK_S lag. The hold sways sway rad on the
    first held joint and the left shoulder roll, so the measured pose drifts before
    golem_safety engages. /lowstate goes out at rate_hz from start().
    """

    TRACK_S = 0.05
    SWAY_PERIOD_S = 2.0

    def __init__(self, ros, profile, q0, topic, sway=0.1, rate_hz=500.0):
        """Hang at body q0, reading commands on topic; nothing is published before start()."""
        self.profile = profile
        self.node = ros.node('relayed_robot')
        self.q0 = np.array(q0, dtype=np.float64)
        self.upper = list(profile.upper_slots)
        self.sway_slots = slots_of(profile, (profile.held_joints[0], 'left_shoulder_roll_joint'))
        self._sway = sway
        self._lock = threading.Lock()
        self._q = self.q0.copy()
        self._cmd_q = self.q0.copy()
        self._commanded = np.zeros(len(self.q0), dtype=bool)
        self._push = np.zeros(len(self.q0))
        self._state = _lowstate(profile)
        self.started = None
        self._last = None
        self._state_publisher = self.node.create_publisher(LowState, '/lowstate', WRITER_QOS)
        self.node.create_subscription(LowCmd, topic, self._on_command, RECORD_QOS)
        self._loop = _Loop(rate_hz, self._step)
        ros.on_close(self._loop.stop)

    def start(self):
        """Start publishing /lowstate."""
        self.started = time.monotonic()
        self._last = self.started
        self._loop.start()

    def _on_command(self, msg):
        """Take the upper slots with kp > 0 as commanded, at their q."""
        motors = msg.motor_cmd
        with self._lock:
            for slot in self.upper:
                self._commanded[slot] = motors[slot].kp > 0.0
                if self._commanded[slot]:
                    self._cmd_q[slot] = motors[slot].q

    def _hold(self, t):
        """Return the swaying limp pose at t s after start()."""
        hold = self.q0.copy()
        hold[self.sway_slots] += self._sway * np.sin(2.0 * np.pi * t / self.SWAY_PERIOD_S)
        return hold

    def _step(self):
        """Move each upper slot toward its target by the lag, then publish /lowstate."""
        now = time.monotonic()
        # the lag in real time, however late this step runs
        alpha = 1.0 - np.exp(-(now - self._last) / self.TRACK_S)
        self._last = now
        with self._lock:
            hold = self._hold(now - self.started)
            target = np.where(self._commanded, self._cmd_q, hold) + self._push
            self._q[self.upper] += (target[self.upper] - self._q[self.upper]) * alpha
            q = self._q.copy()
        self._state.tick = int((now - self.started) * 1000.0)
        for slot, value in enumerate(q):
            self._state.motor_state[slot].q = float(value)
        self._state_publisher.publish(self._state)

    def set_push(self, push):
        """Add push (rad per body slot) to every upper slot's target from now on."""
        with self._lock:
            self._push = np.asarray(push, dtype=np.float64).copy()

    def measured(self):
        """Return the robot's body q."""
        with self._lock:
            return self._q.copy()


class CommandRecorder:
    """Record every LowCmd on topic as a Sample.

    A Sample is the receive time (time.monotonic()), the upper-slot q and their smallest kp,
    and the robot's measured upper-slot q at that time (None without a robot).
    """

    def __init__(self, ros, topic, profile, robot=None):
        """Subscribe to topic, keeping every sample; robot gives the measured pose."""
        self.node = ros.node('command_recorder')
        self._upper = list(profile.upper_slots)
        self._robot = robot
        self._lock = threading.Lock()
        self._samples = []
        self.node.create_subscription(LowCmd, topic, self._on_command, RECORD_QOS)

    def _on_command(self, msg):
        """Record msg as a Sample."""
        now = time.monotonic()
        motors = msg.motor_cmd
        q = np.array([motors[slot].q for slot in self._upper], dtype=np.float64)
        kp = min(motors[slot].kp for slot in self._upper)
        measured = None if self._robot is None else self._robot.measured()[self._upper]
        with self._lock:
            self._samples.append(Sample(now, q, kp, measured))

    @property
    def samples(self):
        """Return the Samples so far."""
        with self._lock:
            return list(self._samples)

    @property
    def count(self):
        """Return how many commands arrived."""
        with self._lock:
            return len(self._samples)


class Sample:
    """Hold one recorded command: t (s), q and measured (rad, upper slots) and min kp."""

    __slots__ = ('t', 'q', 'kp', 'measured')

    def __init__(self, t, q, kp, measured):
        """Hold the fields."""
        self.t, self.q, self.kp, self.measured = t, q, kp, measured


def send_goal(ros, action_type, action_name, goal, timeout=60.0):
    """Send goal to the action server and wait for the outcome; return (status, result)."""
    node = ros.node('action_client')
    client = ActionClient(node, action_type, action_name)
    assert client.wait_for_server(timeout_sec=10.0), f'no {action_name} action server'
    sent = client.send_goal_async(goal)
    assert wait_for(sent.done, 10.0), f'{action_name} did not answer the goal'
    handle = sent.result()
    assert handle.accepted, f'{action_name} refused the goal'
    done = handle.get_result_async()
    assert wait_for(done.done, timeout), f'{action_name} gave no result within {timeout} s'
    response = done.result()
    return response.status, response.result


class SafetyNode:
    """Run golem_safety's safety_node as a subprocess on domain 92.

    It runs on a copy of one of its robot's configs with logging off, written to
    directory/<robot>/.
    """

    def __init__(self, robot, config, directory, sim=False):
        """Write the config copy; start() starts the node."""
        from ament_index_python.packages import get_package_prefix, get_package_share_directory
        share = Path(get_package_share_directory('golem_safety'))
        raw = yaml.safe_load((share / 'config' / robot / f'{config}.yaml').read_text())
        raw['logging']['enabled'] = False
        path = Path(directory) / robot / f'{config}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw))
        prefix = Path(get_package_prefix('golem_safety'))
        executable = prefix / 'lib' / 'golem_safety' / 'safety_node'
        self._args = [str(executable), '--ros-args', '-p', f'robot:={robot}',
                      '-p', f'config:={path}', '-p', f'sim:={"true" if sim else "false"}']
        self.lines = []
        self.proc = None
        self.started = None

    def start(self):
        """Start the node, collecting its output."""
        env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN), RCUTILS_COLORIZED_OUTPUT='0')
        self.started = time.monotonic()
        self.proc = subprocess.Popen(self._args, env=env, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        """Collect the node's output lines."""
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def output(self):
        """Return the node's output so far."""
        return '\n'.join(self.lines)

    def estopped(self):
        """Return whether the node logged an estop event."""
        return any('"event": "estop_' in line for line in list(self.lines))

    def stop(self, timeout=10.0):
        """Stop the node with SIGINT (killed after timeout s)."""
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5.0)
