'''
Fixtures for the safety_node tests, on ROS domain 91: the node as a subprocess, and a
fake robot that publishes /lowstate (and the controllers' commands and the hardware
estop) and records /lowcmd, /arm_sdk and /safety/heartbeat with their source timestamps.

rclpy is imported only by these fixtures, so the other pytest files in this directory,
which load this conftest too, do not need a ROS domain.
'''

import os
import re
import copy
import json
import time
import itertools
import threading
import subprocess
from pathlib import Path

import pytest
import yaml

# golem_safety's integration-test domain; never 0, the real robots' command bus
TEST_DOMAIN = 91
STATE_PERIOD = 0.005  # /lowstate at 200 Hz
STREAM_EVERY = 2  # streamed commands and estop samples every 2 state periods: 100 Hz
OUTPUTS = ('/lowcmd', '/arm_sdk')
HEARTBEAT = '/safety/heartbeat'
# the pose the fake robot reports, which the tests command unless they test the gate
KNEE = 3  # left knee on both robots
KNEE_Q = 0.5
SHOULDER_Q = 0.2
# rcutils' default console format: [LEVEL] [seconds] [logger]: message
LOG_LINE = re.compile(r'^\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[([0-9.]+)\] \[[^\]]*\]: (.*)$')
_names = itertools.count()


def share_dir() -> Path:
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('golem_safety'))


def node_executable() -> str:
    from ament_index_python.packages import get_package_prefix
    return str(Path(get_package_prefix('golem_safety')) / 'lib' / 'golem_safety' / 'safety_node')


class Profile:
    '''The robot facts the tests need, from the installed robots/<robot>.yaml'''
    def __init__(self, robot: str) -> None:
        data = yaml.safe_load((share_dir() / 'robots' / f'{robot}.yaml').read_text())
        self.robot = robot
        self.mode_machine = data['mode_machine']
        self.joints = [joint['name'] for joint in data['joints']]
        self.lower = range(data['lower_slots'][0], data['lower_slots'][1] + 1)
        self.upper = range(data['upper_slots'][0], data['upper_slots'][1] + 1)
        self.weight_slot = data.get('arm_sdk_weight_slot')
        self.shoulder = self.joints.index('left_shoulder_pitch_joint')
        # the joints a controller holds stiff: the G1's waist, the H1-2's torso
        self.held = [i for i, name in enumerate(self.joints)
                     if name.startswith('waist_') or name == 'torso_joint']
        self.position = [tuple(joint['position']) for joint in data['joints']]


def node_env(extra: dict | None = None, drop: tuple = ()) -> dict:
    '''The node's environment: domain 91, uncoloured log lines'''
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(TEST_DOMAIN)
    env['RCUTILS_COLORIZED_OUTPUT'] = '0'
    env.pop('RCUTILS_CONSOLE_OUTPUT_FORMAT', None)
    env.pop('RCUTILS_LOGGING_USE_STDOUT', None)
    for key in drop:
        env.pop(key, None)
    env.update(extra or {})
    return env


def parse_events(lines: list[str]) -> list[tuple[str, float, dict]]:
    '''(level, log time, event) of each JSON event the node logged; prose lines are skipped'''
    events = []
    for line in lines:
        match = LOG_LINE.match(line)
        if not match or not match.group(3).startswith('{'):
            continue
        try:
            events.append((match.group(1), float(match.group(2)), json.loads(match.group(3))))
        except json.JSONDecodeError:
            pass
    return events


class SafetyNode:
    '''safety_node as a subprocess; its output lines are collected by a thread'''
    def __init__(self, robot: str, config: str | Path, *, sim: bool = True,
                 use_estop: bool = False, env: dict | None = None,
                 stdout=None, stderr=None, params: dict | None = None) -> None:
        args = [node_executable(), '--ros-args']
        values = {'robot': robot, 'config': str(config),
                  'sim': 'true' if sim else 'false', 'use_estop': 'true' if use_estop else 'false'}
        values.update(params or {})
        for name, value in values.items():
            if value is not None:
                args += ['-p', f'{name}:={value}']
        self.lines: list[str] = []
        self._reader = None
        if stdout is None:
            self.proc = subprocess.Popen(
                args, env=env if env is not None else node_env(), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
        else:
            self.proc = subprocess.Popen(
                args, env=env if env is not None else node_env(), stdout=stdout, stderr=stderr)

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def output(self) -> str:
        return '\n'.join(self.lines)

    def events(self, name: str | None = None) -> list[dict]:
        return [e for _, _, e in parse_events(list(self.lines))
                if name is None or e['event'] == name]

    def timed_events(self, name: str) -> list[tuple[str, float, dict]]:
        '''(level, log time, event) of each event called name'''
        return [entry for entry in parse_events(list(self.lines)) if entry[2]['event'] == name]

    def wait_for_event(self, name: str, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.events(name)
            if found:
                return found[0]
            assert self.proc.poll() is None, f'the node exited:\n{self.output()}'
            time.sleep(0.02)
        raise AssertionError(f'no {name} within {timeout} s:\n{self.output()}')

    def wait_exit(self, timeout: float) -> int:
        code = self.proc.wait(timeout=timeout)
        if self._reader is not None:
            self._reader.join(timeout=5.0)
        return code

    def stop(self, timeout: float = 10.0) -> None:
        '''SIGTERM, so that the node leaves the DDS graph cleanly; SIGKILL if it hangs'''
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self._reader is not None:
            self._reader.join(timeout=5.0)


class FakeRobot:
    '''
    An rclpy node that streams /lowstate at 200 Hz (and, when asked, commands and the
    hardware estop at 100 Hz) from a thread, and records what the safety node writes as
    (source timestamp, message): the node's own publish time on the wall clock, so that
    the timing of the records does not depend on this process's scheduling
    '''
    def __init__(self, robot: str, context, stream_state: bool = True) -> None:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool
        from unitree_hg.msg import LowCmd, LowState
        self._lowcmd_type = LowCmd
        self._bool_type = Bool
        self.profile = Profile(robot)
        self.node = rclpy.create_node(f'fake_robot_{os.getpid()}_{next(_names)}',
                                      context=context)
        self.state = LowState()
        self.state.mode_machine = self.profile.mode_machine
        self.state.motor_state[KNEE].q = KNEE_Q
        self.state.motor_state[self.profile.shoulder].q = SHOULDER_Q
        self.last_state_write: float | None = None
        self._paused = threading.Event()
        if not stream_state:
            self._paused.set()
        self._state_pub = self.node.create_publisher(LowState, '/lowstate', 10)
        self._estop_pubs = (self.node.create_publisher(Bool, '/estop/triggered', 10),
                            self.node.create_publisher(Bool, '/estop/plugged_in', 10))
        # (triggered, plugged_in) streamed when set; None leaves that topic silent
        self.estop_values: tuple | None = None
        self._streamed: list = []  # (publisher, command)
        # written once in place of the first streamed command
        self.once = None
        self._publishers: list = []
        qos = QoSProfile(depth=10000, reliability=ReliabilityPolicy.RELIABLE)
        self._subs = {topic: self.node.create_subscription(LowCmd, topic, lambda m: None, qos)
                      for topic in OUTPUTS}
        self._subs[HEARTBEAT] = self.node.create_subscription(Bool, HEARTBEAT, lambda m: None, qos)
        self.received = {topic: [] for topic in self._subs}
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._stream, daemon=True),
                         threading.Thread(target=self._record, daemon=True)]
        for thread in self._threads:
            thread.start()

    # what the node writes

    def outputs(self, topic: str, since: float | None = None) -> list:
        records = list(self.received[topic])
        return records if since is None else [r for r in records if r[0] >= since]

    def wait_for_output(self, topic: str, check, timeout: float = 10.0) -> None:
        '''Wait until the newest message on topic passes check'''
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            records = self.received[topic]
            if records and check(records[-1][1]):
                return
            time.sleep(0.01)
        raise AssertionError(f'{topic} never carried the expected command')

    def _record(self) -> None:
        while not self._stop.is_set():
            got = False
            for topic, sub in self._subs.items():
                while True:
                    with sub.handle:
                        taken = sub.handle.take_message(sub.msg_type, sub.raw)
                    if taken is None:
                        break
                    msg, info = taken
                    value = msg.data if topic == HEARTBEAT else msg
                    self.received[topic].append((info['source_timestamp'] * 1e-9, value))
                    got = True
            if not got:
                time.sleep(0.001)

    # what the robot and its controllers send

    def command(self, knee: float | None = None, shoulder: float | None = None,
                held_kp: float = 60.0, q: dict | None = None):
        '''
        A controller's command: the held joints at 0 with held_kp, and knee, shoulder and
        each slot of q at its position, all with kp 60 and kd 1.5
        '''
        cmd = self._lowcmd_type()
        for slot in self.profile.held:
            motor = cmd.motor_cmd[slot]
            motor.mode, motor.kp, motor.kd = 1, held_kp, 1.5
        targets = {KNEE: knee, self.profile.shoulder: shoulder}
        targets.update(q or {})
        for slot, position in targets.items():
            if position is not None:
                motor = cmd.motor_cmd[slot]
                motor.mode, motor.q, motor.kp, motor.kd = 1, position, 60.0, 1.5
        return cmd

    def publisher(self, topic: str, wait: bool = True):
        '''A new writer on topic; wait: until the node's reader has matched it'''
        pub = self.node.create_publisher(self._lowcmd_type, topic, 10)
        self._publishers.append(pub)
        if wait:
            deadline = time.monotonic() + 15.0
            while pub.get_subscription_count() < 1:
                assert time.monotonic() < deadline, f'no reader matched {topic}'
                time.sleep(0.01)
            # the reader side matches on the same discovery round; leave it time to
            time.sleep(0.1)
        return pub

    def close_publisher(self, pub) -> None:
        self._streamed = [(p, c) for p, c in self._streamed if p is not pub]
        self._publishers.remove(pub)
        self.node.destroy_publisher(pub)

    def stream(self, *pairs) -> None:
        '''Write each (publisher, command) at 100 Hz from the stream thread until changed'''
        self._streamed = list(pairs)

    def publish_for(self, seconds: float, *pairs, hz: float = 100.0) -> float:
        '''Write each (publisher, command) pair at hz; return the last write's time'''
        period = 1.0 / hz
        deadline = time.monotonic() + seconds
        next_write = time.monotonic()
        last = time.time()
        while time.monotonic() < deadline:
            last = time.time()  # taken before the write: the node receives it no earlier
            for pub, cmd in pairs:
                pub.publish(cmd)
            next_write += period
            time.sleep(max(0.0, next_write - time.monotonic()))
        return last

    def pause(self) -> None:
        self._paused.set()
        time.sleep(0.02)  # let a write in progress finish

    def resume(self) -> None:
        self._paused.clear()

    def _stream(self) -> None:
        tick = 0
        next_write = time.monotonic()
        while not self._stop.is_set():
            if not self._paused.is_set():
                self.state.tick = (self.state.tick + 5) & 0xffffffff
                self.last_state_write = time.time()
                self._state_pub.publish(self.state)
            if tick % STREAM_EVERY == 0:
                values = self.estop_values
                if values is not None:
                    for pub, value in zip(self._estop_pubs, values):
                        if value is not None:
                            pub.publish(self._bool_type(data=value))
                for pub, cmd in list(self._streamed):
                    once, self.once = self.once, None
                    pub.publish(cmd if once is None else once)
            tick += 1
            next_write += STATE_PERIOD
            delay = next_write - time.monotonic()
            if delay < -0.1:  # a long stall: restart the schedule instead of bursting
                next_write = time.monotonic()
            time.sleep(max(0.0, delay))

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self.node.destroy_node()


class Run:
    '''One test's safety node, fake robot and config'''
    def __init__(self, robot: FakeRobot, node: SafetyNode, config: dict, path: Path) -> None:
        self.robot = robot
        self.node = node
        self.config = config
        self.path = path

    @property
    def out_topic(self) -> str:
        return self.config['topics']['low_cmd_out']

    def outputs(self, since: float | None = None) -> list:
        return self.robot.outputs(self.out_topic, since)

    def close(self) -> None:
        self.node.stop()
        self.robot.close()


def write_config(directory: Path, robot: str, name: str, logging: dict | None = None,
                 **sections) -> tuple[Path, dict]:
    '''
    The installed config name of robot, logging off (or as given), with each keyword
    section's keys replaced, written to directory/robot/name.yaml (a directory named after
    the robot, as the node requires of a config path)
    '''
    raw = yaml.safe_load((share_dir() / 'config' / robot / f'{name}.yaml').read_text())
    raw['logging'] = {**raw.get('logging', {}), 'enabled': False, **(logging or {})}
    for section, values in sections.items():
        if isinstance(values, dict):
            raw[section] = dict(raw.get(section, {}), **values)
        else:
            raw[section] = values
    path = directory / robot / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path, raw


@pytest.fixture(scope='session')
def ros_context():
    import rclpy
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=TEST_DOMAIN)
    yield context
    context.try_shutdown()


@pytest.fixture
def fake_robot(ros_context):
    '''fake_robot(robot, stream_state=True) -> FakeRobot, closed after the test'''
    robots = []

    def make(robot: str, stream_state: bool = True) -> FakeRobot:
        made = FakeRobot(robot, ros_context, stream_state)
        robots.append(made)
        return made

    yield make
    for made in robots:
        made.close()


@pytest.fixture
def stack(tmp_path, ros_context):
    '''
    stack(robot, config, ...) -> Run: a fake robot, then the node on a copy of the
    installed config, started and discovered. stale.state_seconds is 1.0 unless the test
    passes state_stale=None (the shipped value) or its own: a stall of this process's
    state thread must not hold an arm_sdk ramp or report a quiet /lowstate in a test of
    something else. Each run is stopped with SIGTERM after the test
    '''
    runs = []

    def start(robot: str, config: str, *, stream_state: bool = True, use_estop: bool = False,
              state_stale: float | None = 1.0, logging: dict | None = None,
              state_q: dict | None = None, **sections) -> Run:
        sections = copy.deepcopy(sections)
        if state_stale is not None:
            sections['stale'] = dict(sections.get('stale', {}), state_seconds=state_stale)
        path, raw = write_config(tmp_path, robot, config, logging, **sections)
        robot_ = FakeRobot(robot, ros_context, stream_state)
        # the measured q of these slots from the first /lowstate on
        for slot, q in (state_q or {}).items():
            robot_.state.motor_state[slot].q = q
        node = SafetyNode(robot, path, use_estop=use_estop)
        run = Run(robot_, node, raw, path)
        runs.append(run)
        node.wait_for_event('relay_started')
        deadline = time.monotonic() + 15.0
        # the node's /lowstate reader and its output writer
        while (robot_._state_pub.get_subscription_count() < 1
               or robot_.node.count_publishers(run.out_topic) < 1):
            assert time.monotonic() < deadline, 'the node was never discovered'
            time.sleep(0.01)
        return run

    yield start
    for run in runs:
        run.close()
