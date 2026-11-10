"""
Test safety_node over ROS on domain 91, against a fake robot.

Covers the start-up refusals, the relay, the zero-gain start-up, the estop latch, the
heartbeat, the hardware estop and the exit, each for the robots whose behaviour it pins, and
robot time: under use_sim_time the relay and the stale times count a /clock that runs at
other speeds than the wall clock, and the exit keeps the boot clock.
Timing is read from the node's own publish times (the source timestamps), and assertions on
time keep generous margins.
"""

import os
import json
import functools
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import (
    HEARTBEAT, KNEE, KNEE_Q, SHOULDER_Q, Profile, SafetyNode, node_env, share_dir, write_config,
)

UPPER = '/safety/lowcmd_upper_in'
LOWER = '/safety/lowcmd_lower_in'
FULL_IN = '/safety/lowcmd_in'
SPLIT = {'h12': 'sim_safety_split', 'g1': 'sim_split'}
FULL = {'h12': 'default_safety_full', 'g1': 'full'}
# s: long enough for every input of the mode to engage and its commands to be relayed.
SETTLE = 0.9
FIXTURES = Path(__file__).resolve().parent / 'fixtures'


@functools.lru_cache(maxsize=None)
def profile(robot: str) -> Profile:
    """Return the cached profile facts of `robot`."""
    return Profile(robot)


def is_zeroed(msg, joints: int) -> bool:
    """Return whether every body motor is off: the zero-gain stream and the zero estop command."""
    return all(m.mode == 0 and m.q == m.dq == m.tau == m.kp == m.kd == 0.0
               for m in msg.motor_cmd[:joints])


def is_damped(msg, kd: list) -> bool:
    """Return whether every body motor is on with kp 0 and only kd: the damp estop command."""
    return all(m.mode == 1 and m.q == m.dq == m.tau == m.kp == 0.0 and abs(m.kd - kd[i]) < 1e-6
               for i, m in enumerate(msg.motor_cmd[:len(kd)]))


def close(a: float, b: float) -> bool:
    """Return whether a and b agree to 1e-6."""
    return abs(a - b) < 1e-6


def reasons(node: SafetyNode, event: str = 'estop_command_written') -> list[str]:
    """Return the reasons of the node's `event` events."""
    return [e['reason'] for e in node.events(event)]


def estop_events(node: SafetyNode) -> list[dict]:
    """Return the node's events whose name starts with estop."""
    return [e for e in node.events() if e['event'].startswith('estop')]


def input_pairs(run, knee: float = KNEE_Q, shoulder: float = SHOULDER_Q) -> list:
    """Return a matched publisher and the controller's command for each input of the mode."""
    robot, mode = run.robot, run.config['mode']
    if mode == 'split_mode':
        return [(robot.publisher(LOWER), robot.command(knee=knee)),
                (robot.publisher(UPPER), robot.command(shoulder=shoulder))]
    assert mode == 'full_body_mode', mode
    return [(robot.publisher(FULL_IN), robot.command(knee=knee, shoulder=shoulder))]


def carries(msg, run, knee: float = KNEE_Q, shoulder: float = SHOULDER_Q) -> bool:
    """Return whether `msg` carries the commanded knee and shoulder."""
    return close(msg.motor_cmd[KNEE].q, knee) and close(
        msg.motor_cmd[run.robot.profile.shoulder].q, shoulder)


def send_once(robot, pub, cmd, streamed: list) -> None:
    """Write `cmd` once on `pub` with the stream paused, so no streamed command overwrites it."""
    robot.stream()
    time.sleep(0.03)
    pub.publish(cmd)
    time.sleep(0.05)
    robot.stream(*streamed)


# The Unitree LowCmd CRC, written from its definition, to check the node's independently.

def _crc_table() -> list[int]:
    """Return the CRC-32 table of polynomial 0x04c11db7, MSB first."""
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else crc << 1
            crc &= 0xFFFFFFFF
        table.append(crc)
    return table


CRC_TABLE = _crc_table()


def lowcmd_bytes(mode_pr, mode_machine, motors, reserve) -> bytes:
    """Return the C struct the robot checksums; motors are (mode, q, dq, tau, kp, kd, reserve)."""
    data = struct.pack('<BBxx', mode_pr, mode_machine)
    for motor in motors:
        data += struct.pack('<Bxxx5fI', *motor)
    return data + struct.pack('<4I', *reserve)


def unitree_crc(data: bytes) -> int:
    """Return CRC-32 (0x04c11db7, init 0xffffffff, no reflection) over each 32-bit word."""
    crc = 0xFFFFFFFF
    for (word,) in struct.iter_unpack('<I', data):
        for byte in word.to_bytes(4, 'big'):
            crc = ((crc << 8) & 0xFFFFFFFF) ^ CRC_TABLE[(crc >> 24) ^ byte]
    return crc


def msg_crc(msg) -> int:
    """Return the Unitree CRC of a LowCmd message."""
    motors = [(m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve) for m in msg.motor_cmd]
    return unitree_crc(lowcmd_bytes(msg.mode_pr, msg.mode_machine, motors, list(msg.reserve)))


def test_python_crc_matches_the_sdk_fixtures():
    """This file's CRC gives the Unitree SDK's CRC for every recorded command."""
    fixtures = json.loads((FIXTURES / 'lowcmd_crc.json').read_text())['commands']
    for c in fixtures:
        motors = list(zip(c['mode'], c['q'], c['dq'], c['tau'], c['kp'], c['kd'],
                          c['motor_reserve']))
        data = lowcmd_bytes(c['mode_pr'], c['mode_machine'], motors, c['reserve'])
        assert unitree_crc(data) == c['crc'], c['name']


def _refused(robot, config, timeout: float = 30.0, **kwargs) -> tuple[int, str]:
    """Run the node until it exits; return its exit code and output."""
    node = SafetyNode(robot, config, **kwargs)
    try:
        code = node.wait_exit(timeout)
    finally:
        node.stop()
    return code, node.output()


@pytest.mark.parametrize('robot, config, words', [
    ('h12', 'split', ['h12', 'split', "no config 'split' for robot h12"]),
    ('g1', 'default_safety_split',
     ['g1', 'default_safety_split', "no config 'default_safety_split' for robot g1"]),
    ('h12', 'share:g1/full.yaml',
     ['h12', 'g1/full.yaml', 'in the config directory of robot g1, not of robot h12']),
    ('g1', 'share:h12/sim_safety_split.yaml',
     ['g1', 'h12/sim_safety_split.yaml', 'in the config directory of robot h12, not of robot g1']),
], ids=['h12-g1_name', 'g1-h12_name', 'h12-g1_path', 'g1-h12_path'])
def test_config_for_other_robot_is_refused(robot, config, words):
    """SAFETY: another robot's config, by name or by path, is refused before relaying."""
    if config.startswith('share:'):
        config = str(share_dir() / 'config' / config.split(':', 1)[1])
    code, output = _refused(robot, config)
    assert code != 0, output
    refusal = [line for line in output.splitlines() if 'refuses to start' in line]
    assert len(refusal) == 1, output
    for word in words:
        assert word in refusal[0], refusal[0]
    assert 'relay_started' not in output


def test_a_config_that_fails_to_load_is_refused(tmp_path):
    """A config that does not load is refused with the loader's reason."""
    path, _ = write_config(tmp_path, 'g1', 'split', mode='no_such_mode')
    code, output = _refused('g1', path)
    assert code != 0, output
    assert "'no_such_mode' is not a mode of robot g1 (split_mode, full_body_mode)" in output


@pytest.mark.parametrize('params, word', [
    ({'robot': None}, 'robot'),
    ({'config': None}, 'config'),
    ({'robot': 'h13'}, 'h13'),
])
def test_missing_or_unknown_parameters_are_refused(params, word):
    """A missing robot or config, or an unknown robot, is refused."""
    node = SafetyNode('h12', 'sim_safety_split', params=params)
    try:
        code = node.wait_exit(30.0)
    finally:
        node.stop()
    output = node.output()
    assert code != 0, output
    refusal = [line for line in output.splitlines() if 'refuses to start' in line]
    assert len(refusal) == 1 and word in refusal[0], output


# A passive watcher on domain 0, loopback only: it prints what it sees of /lowcmd.
DOMAIN_0_WATCHER = '''
import sys, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_hg.msg import LowCmd
rclpy.init()
node = rclpy.create_node('domain_0_watcher')
got = []
qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
node.create_subscription(LowCmd, '/lowcmd', got.append, qos)
print('ready', flush=True)
seen = 0
deadline = time.monotonic() + float(sys.argv[1])
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.01)
    seen = max(seen, node.count_publishers('/lowcmd'))
print(f'publishers {seen} messages {len(got)}', flush=True)
'''


# SAFETY: CycloneDDS on the loopback interface only, with unicast discovery between the
# processes of this host, so nothing on domain 0 here can reach a real robot.
LOOPBACK_ONLY = (
    '<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="lo"/>'
    '</Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery>'
    '<ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>100'
    '</MaxAutoParticipantIndex><Peers><Peer address="127.0.0.1"/></Peers></Discovery>'
    '</Domain></CycloneDDS>')


@pytest.mark.parametrize('domain', ['0', None], ids=['ROS_DOMAIN_ID=0', 'ROS_DOMAIN_ID_unset'])
def test_sim_refuses_domain_zero(domain):
    """SAFETY: sim:=true on domain 0, given or by default, is refused before it publishes."""
    extra = {'CYCLONEDDS_URI': LOOPBACK_ONLY}
    if domain is not None:
        extra['ROS_DOMAIN_ID'] = domain
    drop = ('ROS_LOCALHOST_ONLY',) + (('ROS_DOMAIN_ID',) if domain is None else ())
    env = node_env(extra, drop=drop)
    # SAFETY: CYCLONEDDS_URI, which keeps domain 0 on the loopback interface, is read by
    # CycloneDDS only: under another RMW the processes below would join domain 0 on the real
    # interfaces.
    assert env.get('RMW_IMPLEMENTATION') == 'rmw_cyclonedds_cpp', (
        'this test needs RMW_IMPLEMENTATION=rmw_cyclonedds_cpp: under any other RMW the '
        'loopback-only CYCLONEDDS_URI has no effect, and the test would join domain 0, the '
        "real robots' command bus, on the real network interfaces")
    watcher = subprocess.Popen([sys.executable, '-c', DOMAIN_0_WATCHER, '4.0'], env=env,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert watcher.stdout.readline().strip() == 'ready'
        node = SafetyNode('h12', 'sim_safety_split', sim=True, env=env)
        try:
            code = node.wait_exit(30.0)
        finally:
            node.stop()
        seen = watcher.communicate(timeout=30.0)[0]
    finally:
        if watcher.poll() is None:
            watcher.kill()
            watcher.wait()
    output = node.output()
    assert code != 0, output
    refusal = [line for line in output.splitlines() if 'refuses to start' in line]
    assert len(refusal) == 1 and 'sim' in refusal[0] and 'domain 0' in refusal[0], output
    assert 'relay_started' not in output
    assert 'publishers 0 messages 0' in seen, seen


def test_use_sim_time_needs_sim():
    """SAFETY: use_sim_time:=true without sim:=true is refused: real robots keep the boot clock."""
    node = SafetyNode('h12', 'sim_safety_split', sim=False, params={'use_sim_time': 'true'})
    try:
        code = node.wait_exit(30.0)
    finally:
        node.stop()
    output = node.output()
    assert code != 0, output
    refusal = [line for line in output.splitlines() if 'refuses to start' in line]
    assert len(refusal) == 1, output
    assert 'use_sim_time:=true needs sim:=true' in refusal[0], output
    assert 'relay_started' not in output


SIM_TIME = {'use_sim_time': 'true'}


@pytest.mark.parametrize('use_sim_time, per_wall_s', [(True, 250), (False, 500)],
                         ids=['sim_time', 'boot_clock'])
def test_relay_rate_counts_robot_time(stack, sim_clock, use_sim_time, per_wall_s):
    """
    The relay writes control.publish_hz per s of robot time: of /clock under use_sim_time, of
    the boot clock otherwise; on /clock it starts with the first one and stops while it does.
    """
    clock = sim_clock(rate=0.5)
    run = stack('g1', 'sim_split', params=SIM_TIME if use_sim_time else None)
    time.sleep(0.5)
    if use_sim_time:
        assert run.outputs() == [], 'relayed before the first /clock'
        assert 'waiting for the first /clock' in run.node.output()
    clock.start()
    time.sleep(0.5)
    start = time.time()
    time.sleep(1.0)
    end = time.time()
    written = len([t for t, _ in run.outputs(since=start) if t < end])
    # publish_hz 500 per s of robot time: 250 per wall s with /clock at half speed
    assert 0.85 * per_wall_s < written < 1.1 * per_wall_s, written
    clock.pause()
    time.sleep(0.1)
    paused = time.time()
    time.sleep(0.4)
    stalled = len(run.outputs(since=paused))
    assert (stalled == 0) if use_sim_time else (stalled > 150), stalled


def test_quiet_input_counts_sim_time(stack, sim_clock):
    """Under use_sim_time an input is reported quiet after stale seconds of sim time."""
    seconds = 0.2
    clock = sim_clock(rate=0.25)
    clock.start()
    run = stack('g1', 'sim_split', stale={'lower_seconds': seconds}, params=SIM_TIME)
    pairs = input_pairs(run)
    run.robot.stream(*pairs)
    run.robot.wait_for_output('/lowcmd', lambda msg: carries(msg, run), timeout=15.0)
    time.sleep(0.5)
    run.robot.stream(*[(pub, cmd) for pub, cmd in pairs if pub.topic_name != LOWER])
    stopped = time.monotonic()
    event = run.node.wait_for_event('input_quiet', timeout=10.0)
    waited = time.monotonic() - stopped
    # 0.2 s of sim time: 0.8 s of wall time at a quarter speed
    assert waited > 0.6, waited
    assert event['topic'] == LOWER and event['seconds'] == seconds
    assert seconds <= event['age'] < seconds + 0.1, event
    assert estop_events(run.node) == []


def test_sim_time_exit_finishes_the_burst_with_clock_stopped(stack, sim_clock):
    """
    SAFETY: under use_sim_time the exit sequence runs on the boot clock: with /clock stopped
    mid-burst, a SIGTERM still writes the rest of the H1-2's estop burst and exits 0.
    """
    clock = sim_clock(rate=0.1)
    clock.start()
    run = stack('h12', 'sim_safety_split', params=SIM_TIME)
    pairs = input_pairs(run)
    run.robot.stream(*pairs)
    run.robot.wait_for_output('/lowcmd', lambda msg: carries(msg, run), timeout=15.0)
    second = run.robot.publisher(UPPER)
    # the first upper writer stops, so the next upper command the relay takes is the second
    # writer's, 0.5 s of sim time (5 s of wall time) within the first's last: a latch. At a
    # tenth of the wall clock's speed the burst of 50 then takes 1 s of wall time
    run.robot.stream(*[(pub, cmd) for pub, cmd in pairs if pub.topic_name != UPPER])
    run.robot.publish_for(0.1, (second, run.robot.command(shoulder=SHOULDER_Q + 0.05)))
    run.node.wait_for_event('estop_command_written', timeout=5.0)
    clock.pause()
    time.sleep(0.1)
    sent = time.time()
    run.node.proc.send_signal(signal.SIGTERM)
    code = run.node.wait_exit(5.0)
    assert code == 0, run.node.output()
    assert time.time() - sent < 1.5
    time.sleep(0.2)
    joints = len(run.robot.profile.joints)
    written = run.outputs()
    first = next(i for i, (_, msg) in enumerate(written) if carries(msg, run))
    burst = [(t, msg) for t, msg in written[first:] if not carries(msg, run)]
    assert len(burst) == 50 and all(is_zeroed(msg, joints) for _, msg in burst), len(burst)
    # the rest of the burst, after the SIGTERM, at publish_hz (500) of the boot clock
    rest = [t for t, _ in burst if t > sent]
    gaps = sorted(b - a for a, b in zip(rest, rest[1:]))
    assert len(rest) > 10 and gaps[len(gaps) // 2] > 0.0015, gaps


def test_the_launch_file_starts_the_node_after_start_delay():
    """safety.launch.py starts one safety_node after start_delay, with the launch arguments."""
    import importlib.util
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, TimerAction
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters
    spec = importlib.util.spec_from_file_location(
        'safety_launch', share_dir() / 'launch' / 'safety.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    declared = {entity.name: entity for entity in description.entities
                if isinstance(entity, DeclareLaunchArgument)}
    assert set(declared) == {'robot', 'config', 'sim', 'use_estop', 'use_sim_time',
                             'start_delay'}
    assert declared['robot'].default_value is None
    assert declared['config'].default_value is None
    context = LaunchContext()
    context.launch_configurations.update(robot='g1', config='sim_split', sim='true',
                                         use_sim_time='true', start_delay='2.5')
    started = []

    def walk(entities, delay):
        """Collect each Node under `entities`, with its total TimerAction delay."""
        for entity in entities:
            if isinstance(entity, Node):
                started.append((delay, entity))
            elif isinstance(entity, TimerAction):
                walk(entity.actions, delay + float(entity.period))
            else:
                walk(entity.visit(context) or [], delay)

    walk(description.entities, 0.0)
    ((delay, node),) = started
    assert (node.node_package, node.node_executable, node._Node__node_name, delay) == (
        'golem_safety', 'safety_node', 'safety_node', 2.5)
    parameters = {}
    for params in evaluate_parameters(context, node._Node__parameters):
        parameters.update(params)
    assert parameters == {'robot': 'g1', 'config': 'sim_split', 'sim': True,
                          'use_estop': False, 'use_sim_time': True}


def test_the_launch_file_runs_the_node():
    """ros2 launch of safety.launch.py runs the node, which exits cleanly on SIGINT."""
    command = ['ros2', 'launch', 'golem_safety', 'safety.launch.py', 'robot:=h12',
               'config:=sim_safety_split', 'sim:=true', 'use_sim_time:=false',
               'start_delay:=0.5']
    launch = subprocess.Popen(command, env=node_env(), text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, start_new_session=True)
    lines = []

    def read() -> None:
        """Collect the launch's output lines."""
        for line in launch.stdout:
            lines.append(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 30.0
        while not any('relay_started' in line for line in lines):
            assert time.monotonic() < deadline and launch.poll() is None, ''.join(lines)
            time.sleep(0.05)
        launch.send_signal(signal.SIGINT)
        launch.wait(timeout=20.0)
    finally:
        if launch.poll() is None:
            os.killpg(launch.pid, signal.SIGKILL)
            launch.wait()
        reader.join(timeout=5.0)
    output = ''.join(lines)
    assert '"robot": "h12"' in output and '"sim": true' in output, output
    assert 'Exit requested (SIGINT)' in output, output
    assert 'process has finished cleanly' in output, output


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_split_relay_end_to_end(stack, robot):
    """split_mode relays each input on its own slots, under the robot's header and a valid CRC."""
    run = stack(robot, SPLIT[robot])
    facts = run.robot.profile
    (lower_pub, lower), (upper_pub, upper) = input_pairs(run)
    lower.motor_cmd[facts.shoulder].q = 0.9  # ignored: the arms come from the upper input
    upper.motor_cmd[KNEE].q = 0.9  # ignored: the legs come from the lower input
    # The header is the profile's, not a copy of an input's.
    lower.mode_machine = 0
    upper.mode_machine = 6 if robot == 'g1' else 5
    run.robot.publish_for(0.5, (lower_pub, lower), (upper_pub, upper))
    time.sleep(0.05)
    tail = [msg for _, msg in run.outputs()][-20:]
    assert len(tail) == 20
    for msg in tail:
        assert (msg.mode_pr, msg.mode_machine) == (0, facts.mode_machine)
        assert carries(msg, run)
        assert all(msg.motor_cmd[i].kp == 60.0 for i in facts.held)
        assert msg.crc == msg_crc(msg)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_newest_only_under_load(stack, robot, record_testsuite_property):
    """
    Under a 2 kHz input the relay keeps 500 Hz, in order, and never relays a command older than
    one period.
    """
    run = stack(robot, SPLIT[robot])
    slot = run.robot.profile.shoulder
    pub = run.robot.publisher(UPPER)
    cmd = run.robot.command(shoulder=SHOULDER_Q)
    # Engage on the measured pose first.
    run.robot.publish_for(0.5, (pub, cmd))
    step = 0.0002  # rad per command: 4000 commands stay inside the shoulder's clip range
    period = 1.0 / run.config['control']['publish_hz']
    published = []  # (time the publish returned, seq)
    start = time.time()
    next_write = time.monotonic()
    deadline = next_write + 2.0
    seq = 0
    while time.monotonic() < deadline:
        seq += 1
        cmd.motor_cmd[slot].q = SHOULDER_Q + seq * step
        pub.publish(cmd)
        published.append((time.time(), seq))
        next_write += 0.0005
        time.sleep(max(0.0, next_write - time.monotonic()))
    end = time.time()
    time.sleep(0.1)
    send_hz = len(published) / (end - start)
    assert send_hz > 1900, f'the load ran at {send_hz:.0f} Hz'

    outputs = [(t, round((msg.motor_cmd[slot].q - SHOULDER_Q) / step))
               for t, msg in run.outputs(since=start + 0.01) if t <= end]
    relay_hz = len(outputs) / (end - start - 0.01)
    record_testsuite_property(f'{robot}_relay_hz', round(relay_hz, 1))
    times = {seq: t for t, seq in published}
    # How long each relayed command had already been superseded when it went out.
    superseded = [t - times[k + 1] for t, k in outputs if k + 1 in times]
    worst = max(superseded)
    record_testsuite_property(f'{robot}_worst_superseded_ms', round(worst * 1000.0, 3))
    assert relay_hz >= 495.0, f'relay at {relay_hz:.1f} Hz'
    assert worst <= period, f'a relayed command was {worst * 1000:.2f} ms old'
    assert [k for _, k in outputs] == sorted(k for _, k in outputs), 'relayed out of order'


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_every_output_carries_the_crc(stack, robot):
    """Every published command carries the Unitree CRC of its fields."""
    run = stack(robot, SPLIT[robot])
    pairs = input_pairs(run)
    pairs[0][1].motor_cmd[KNEE].dq = 0.01
    pairs[1][1].motor_cmd[run.robot.profile.shoulder].tau = 0.5
    run.robot.publish_for(0.5, *pairs)
    time.sleep(0.05)
    published = [msg for _, msg in run.outputs()]
    assert len(published) > 100
    assert any(carries(msg, run) for msg in published)
    for msg in published:
        assert msg.crc == msg_crc(msg)


@pytest.mark.parametrize('first', ['lower', 'upper'])
def test_zero_stream_engages_each_input_on_its_own(stack, first):
    """
    SAFETY: the zero stream runs from boot, and each input engages on its own first command,
    the other input's slots staying at the zero command.
    """
    run = stack('h12', 'sim_safety_split')
    joints = len(run.robot.profile.joints)
    time.sleep(0.3)
    boot = [msg for _, msg in run.outputs()]
    assert boot and all(is_zeroed(msg, joints) and msg.mode_machine == 6 for msg in boot)

    (lower_pub, lower), (upper_pub, upper) = input_pairs(run)
    alone = (lower_pub, lower) if first == 'lower' else (upper_pub, upper)
    since = run.robot.publish_for(0.3, alone)
    time.sleep(0.05)
    shoulder = run.robot.profile.shoulder
    for _, msg in run.outputs(since=since)[-10:]:
        own = KNEE if first == 'lower' else shoulder
        other = run.robot.profile.upper if first == 'lower' else run.robot.profile.lower
        assert msg.motor_cmd[own].kp == 60.0
        assert all(msg.motor_cmd[i].kp == 0.0 and msg.motor_cmd[i].mode == 0 for i in other)
    source = f'split_mode_{first}'
    assert [e['source'] for e in run.node.events('input_engaged')] == [source]

    since = run.robot.publish_for(0.3, (lower_pub, lower), (upper_pub, upper))
    time.sleep(0.05)
    assert all(carries(msg, run) for _, msg in run.outputs(since=since))
    assert sorted(e['source'] for e in run.node.events('input_engaged')) == [
        'split_mode_lower', 'split_mode_upper']


def test_g1_streams_lowcmd_from_boot(stack):
    """
    SAFETY: with no input and no /lowstate the G1 gets the zero stream at about 500 Hz on every
    slot, and no heartbeat, since the relay cannot begin.
    """
    run = stack('g1', 'split', stream_state=False)
    time.sleep(0.3)
    start = time.time()
    time.sleep(1.0)
    end = time.time()
    window = [msg for t, msg in run.outputs(since=start) if t < end]
    hz = len(window) / (end - start)
    assert 450.0 <= hz <= 520.0, f'{hz:.0f} Hz'
    for msg in window:
        assert is_zeroed(msg, len(msg.motor_cmd))
        assert (msg.mode_pr, msg.mode_machine) == (0, 5) and msg.crc == msg_crc(msg)
    assert run.robot.outputs(HEARTBEAT) == []
    assert run.node.events('input_engaged') == [] and estop_events(run.node) == []


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_full_body_mode_clips_gains_and_holds_a_quiet_input(stack, robot):
    """
    full_body_mode clips gains to the config's maxima, reports each once, and holds a quiet
    input's last command while reporting it.
    """
    run = stack(robot, FULL[robot])
    limits = run.config['limits']['clip']
    kp_max, kd_max = limits['kp_max'], limits['kd_max']
    joints = len(run.robot.profile.joints)
    ((pub, cmd),) = input_pairs(run)
    cmd.motor_cmd[KNEE].kp = kp_max + 100.0  # clipped and reported once
    cmd.motor_cmd[KNEE].kd = kd_max + 10.0
    time.sleep(0.2)
    before = [msg for _, msg in run.outputs()]
    assert before and all(is_zeroed(msg, joints) for msg in before)

    last = run.robot.publish_for(0.3, (pub, cmd))
    stale = run.config['stale']['lower_seconds']
    time.sleep(stale + 0.3)
    resumed = run.robot.publish_for(0.2, (pub, cmd))
    time.sleep(0.05)

    engaged = [msg for t, msg in run.outputs() if not is_zeroed(msg, joints)]
    assert engaged
    for msg in engaged:
        assert (msg.mode_pr, msg.mode_machine) == (0, run.robot.profile.mode_machine)
        knee = msg.motor_cmd[KNEE]
        assert carries(msg, run) and (knee.kp, knee.kd) == (kp_max, kd_max)
    # Held while quiet: the relay kept writing the last command.
    quiet = [msg for t, msg in run.outputs(since=last + stale + 0.05) if t < resumed]
    assert len(quiet) > 50 and all(carries(msg, run) for msg in quiet)

    joint = run.robot.profile.joints[KNEE]
    clipped = run.node.events('gain_clipped')
    assert [(e['joint'], e['gain'], e['commanded'], e['max'], e['source']) for e in clipped] == [
        (joint, 'kp', kp_max + 100.0, kp_max, 'full_body_mode'),
        (joint, 'kd', kd_max + 10.0, kd_max, 'full_body_mode')]
    assert [level for level, _, _ in run.node.timed_events('gain_clipped')] == ['WARN', 'WARN']
    assert [e['source'] for e in run.node.events('input_engaged')] == ['full_body_mode']
    (level, _, quiet_event), = run.node.timed_events('input_quiet')
    assert level == 'WARN' and quiet_event['seconds'] == stale
    assert quiet_event['source'] == 'full_body_mode' and quiet_event['topic'] == FULL_IN
    assert len(run.node.events('input_resumed')) == 1
    assert estop_events(run.node) == []


# (robot, config, the input that goes quiet).
QUIET_INPUTS = [
    ('h12', 'sim_safety_split', 'lower'), ('h12', 'sim_safety_split', 'upper'),
    ('h12', 'default_safety_full', 'full'), ('g1', 'sim_split', 'lower'),
    ('g1', 'sim_split', 'upper'), ('g1', 'full', 'full'),
]


@pytest.mark.parametrize('robot, config, quiet', QUIET_INPUTS,
                         ids=[f'{r}-{c}-{q}' for r, c, q in QUIET_INPUTS])
def test_quiet_input_is_held_and_reported(stack, robot, config, quiet):
    """
    SAFETY: a quiet input keeps its last command relayed, is reported quiet and resumed, and
    latches nothing.
    """
    # 0.5 s, not a shipped value, so that a report that ignores the config fails.
    seconds = 0.5
    key = 'upper_seconds' if quiet == 'upper' else 'lower_seconds'
    run = stack(robot, config, stale={key: seconds})
    pairs = input_pairs(run)
    topic = {'lower': LOWER, 'upper': UPPER, 'full': FULL_IN}[quiet]
    names = {LOWER: 'lower', UPPER: 'upper', FULL_IN: 'full'}
    others = [(pub, cmd) for pub, cmd in pairs if names[pub.topic_name] != quiet]
    run.robot.stream(*pairs)
    time.sleep(0.6)
    run.robot.stream(*others)
    stopped = time.time()
    time.sleep(seconds + 0.6)
    run.robot.stream(*pairs)
    resumed = time.time()
    time.sleep(0.3)

    held = [msg for t, msg in run.outputs(since=stopped + seconds + 0.1) if t < resumed]
    assert len(held) > 100 and all(carries(msg, run) for msg in held)
    (level, _, event), = run.node.timed_events('input_quiet')
    assert level == 'WARN'
    assert event['seconds'] == seconds and event['topic'] == topic
    assert seconds <= event['age'] < seconds + 0.5  # rounded to 4 decimals
    (resumed_event,) = run.node.events('input_resumed')
    assert resumed_event['topic'] == topic and resumed_event['gap'] > seconds
    assert estop_events(run.node) == []
    assert run.robot.outputs(HEARTBEAT) and all(beat for _, beat in run.robot.outputs(HEARTBEAT))


QUIET_STATE = [('h12', 'sim_safety_split'), ('g1', 'sim_split'), ('g1', 'full')]


@pytest.mark.parametrize('robot, config', QUIET_STATE, ids=[f'{r}-{c}' for r, c in QUIET_STATE])
def test_quiet_lowstate_is_held_and_reported(stack, robot, config):
    """
    SAFETY: a quiet /lowstate leaves the relay running, is reported quiet and resumed, and
    latches nothing.
    """
    seconds = 0.3
    run = stack(robot, config, state_stale=seconds)
    run.robot.stream(*input_pairs(run))
    time.sleep(SETTLE)
    run.robot.pause()
    paused = run.robot.last_state_write
    time.sleep(seconds + 0.5)
    run.robot.resume()
    resumed = time.time()
    time.sleep(0.3)

    held = [msg for t, msg in run.outputs(since=paused + seconds + 0.1) if t < resumed]
    assert len(held) > 100 and all(carries(msg, run) for msg in held)
    (level, _, event), = run.node.timed_events('state_quiet')
    assert level == 'WARN' and event['seconds'] == seconds and event['topic'] == '/lowstate'
    (resumed_event,) = run.node.events('state_resumed')
    assert resumed_event['gap'] > seconds
    assert estop_events(run.node) == []


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_engage_refusals_are_logged_at_most_once_a_second(stack, robot):
    """
    On another robot's /lowstate no input engages, and engage_refused is logged at most once a
    second.
    """
    run = stack(robot, 'split' if robot == 'g1' else 'sim_safety_split')
    foreign = 6 if robot == 'g1' else 5
    run.robot.state.mode_machine = foreign
    time.sleep(0.1)
    pairs = input_pairs(run)
    # About 250 refused commands per input.
    run.robot.publish_for(2.5, *pairs)
    time.sleep(0.05)
    refused = run.node.timed_events('engage_refused')
    assert len(refused) == 3, refused  # at about 0, 1 and 2 s
    assert all(b[1] - a[1] > 0.9 for a, b in zip(refused, refused[1:]))
    expected = run.robot.profile.mode_machine
    assert {e['reason'] for _, _, e in refused} == {
        f'/lowstate mode_machine is {foreign}, expected {expected}'}
    joints = len(run.robot.profile.joints)
    published = run.outputs()
    assert published and all(is_zeroed(msg, joints) for _, msg in published)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_an_estop_before_engaging(stack, robot):
    """
    SAFETY: a non-finite first command latches before engaging: no input engages, the estop
    command follows, and the heartbeat is false.
    """
    run = stack(robot, 'split' if robot == 'g1' else 'sim_safety_split')
    joints = len(run.robot.profile.joints)
    (lower_pub, lower), (upper_pub, upper) = input_pairs(run)
    # An input engages on its first valid command, so the NaN comes first.
    bad = run.robot.command(knee=float('nan'))
    sent = time.time()
    lower_pub.publish(bad)
    time.sleep(0.05)
    run.robot.publish_for(0.1, (lower_pub, lower))
    run.robot.publish_for(0.3, (lower_pub, lower), (upper_pub, upper))
    time.sleep(0.3)
    reason_prefix = 'Command validation failed on /safety/lowcmd_lower_in: motor 3'
    # The zero stream, then the estop command: the H1-2's burst of 50 and then silence, the
    # G1's every cycle.
    published = run.outputs()
    assert published and all(is_zeroed(msg, joints) for _, msg in published)
    if robot == 'g1':
        assert published[-1][0] > time.time() - 0.1, 'keep_writing'
    else:
        assert 50 <= len(run.outputs(since=sent)) <= 53
        assert published[-1][0] < time.time() - 0.2
    assert run.node.events('input_engaged') == []
    (reason,) = reasons(run.node)
    assert reason.startswith(reason_prefix), reason
    assert run.robot.outputs(HEARTBEAT)[-1][1] is False


DAMP = [('g1', 'split'), ('g1', 'full'), ('h12', 'sim_safety_split')]


@pytest.mark.parametrize('robot, config', DAMP, ids=[f'{r}-{c}' for r, c in DAMP])
def test_damp_estop_keeps_every_motor_on_with_only_kd(stack, robot, config):
    """
    Damp writes every body motor on with kp 0 and the configured kd, as a burst on the H1-2
    and every cycle on the G1.
    """
    joints = len(profile(robot).joints)
    kd = [round(0.1 * (i + 1), 1) for i in range(joints)]  # per joint, under kd_max
    run = stack(robot, config, estop_response={'mode': 'damp', 'kd': kd})
    pairs = input_pairs(run)
    run.robot.stream(*pairs)
    time.sleep(0.3)
    # A NaN on the leg input latches the estop.
    send_once(run.robot, pairs[0][0], run.robot.command(knee=float('nan')), pairs)
    time.sleep(0.3)
    published = run.outputs()
    idx = next(i for i, (_, msg) in enumerate(published) if is_damped(msg, kd))
    assert idx > 0 and carries(published[idx - 1][1], run), 'relayed until the estop'
    after = published[idx:]
    assert all(is_damped(msg, kd) for _, msg in after), 'the estop must latch'
    assert all((msg.mode_pr, msg.mode_machine) == (0, run.robot.profile.mode_machine)
               for _, msg in after)
    if robot == 'h12':
        assert len(after) == 50, 'burst_then_silence: 50 estop commands'
    else:
        assert after[-1][0] > time.time() - 0.1, 'keep_writing'
    (reason,) = reasons(run.node)
    assert reason.startswith('Command validation failed')


STATE_LIMITS = [('g1', 'split'), ('g1', 'full'), ('h12', 'sim_safety_split')]


@pytest.mark.parametrize('robot, config', STATE_LIMITS,
                         ids=[f'{r}-{c}' for r, c in STATE_LIMITS])
def test_state_limits(stack, robot, config):
    """SAFETY: a joint past its estop limit at start-up latches before any input engages."""
    facts = profile(robot)
    # A limp knee resting far past its limit while the layer starts.
    past_limit = facts.position[KNEE][0] - 0.2
    run = stack(robot, config, state_q={KNEE: past_limit})
    joints = len(facts.joints)
    if robot == 'h12':
        # Checked from the first /lowstate; the H1-2 writes its burst, then nothing.
        time.sleep(0.3)
        published = run.outputs()
        assert published and all(is_zeroed(msg, joints) for _, msg in published)
        assert published[-1][0] < time.time() - 0.1, 'the burst, then silence'
        (reason,) = reasons(run.node)
        assert reason.startswith('motor 3 (left_knee_joint) q out of estop range: -0.32 (limit')
        return

    # Checked from the first /lowstate; the G1 keeps writing the estop command, every motor
    # off.
    time.sleep(0.3)
    published = run.outputs()
    assert published and all(is_zeroed(msg, joints) for _, msg in published)
    assert published[-1][0] > time.time() - 0.1, 'keep_writing'
    assert [e['event'] for e in estop_events(run.node)] == ['estop_command_written']
    (reason,) = reasons(run.node)
    assert reason.startswith('motor 3 (left_knee_joint) q out of estop range: '), reason
    assert run.node.events('input_engaged') == []


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_a_lowstate_from_another_robot_after_engaging_latches(stack, robot):
    """SAFETY: once engaged, another robot's /lowstate latches the estop."""
    run = stack(robot, SPLIT[robot])
    run.robot.stream(*input_pairs(run))
    time.sleep(0.4)
    assert estop_events(run.node) == []
    foreign = 6 if robot == 'g1' else 5
    run.robot.state.mode_machine = foreign
    time.sleep(0.3)
    (reason,) = reasons(run.node)
    own = run.robot.profile.mode_machine
    assert reason == (f'/lowstate mode_machine {foreign} is not the {robot}\'s {own}: another '
                      'robot or sim is on this domain')
    joints = len(run.robot.profile.joints)
    assert is_zeroed(run.outputs()[-1][1], joints)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_a_second_writer_latches_the_estop_before_its_command_is_used(stack, robot):
    """SAFETY: a second live writer latches before its command reaches /lowcmd."""
    run = stack(robot, SPLIT[robot])
    pairs = input_pairs(run)
    run.robot.stream(*pairs)
    time.sleep(SETTLE)
    second = run.robot.publisher(UPPER)
    run.robot.publish_for(0.2, (second, run.robot.command(shoulder=SHOULDER_Q + 0.05)))
    time.sleep(SETTLE)
    (reason,) = reasons(run.node)
    assert reason.startswith(f'second writer on {UPPER}: a new writer sent a command '), reason
    shoulder = run.robot.profile.shoulder
    # The second writer's target never reached the output.
    assert not any(close(msg.motor_cmd[shoulder].q, SHOULDER_Q + 0.05) for _, msg in run.outputs())
    assert is_zeroed(run.outputs()[-1][1], len(run.robot.profile.joints))


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_a_restarted_writer_is_accepted_after_a_quiet_gap(stack, robot):
    """A controller restarted after handover_seconds of quiet is accepted as the input's writer."""
    run = stack(robot, SPLIT[robot])
    pairs = input_pairs(run)
    first = pairs[-1][0]  # the upper input's writer
    run.robot.stream(*pairs)
    time.sleep(SETTLE)
    run.robot.stream(*pairs[:-1])
    run.robot.close_publisher(first)
    time.sleep(0.6)  # longer than single_writer.handover_seconds (0.5)
    restarted = run.robot.publisher(UPPER)
    moved = SHOULDER_Q + 0.05
    run.robot.stream(*pairs[:-1], (restarted, run.robot.command(shoulder=moved)))
    time.sleep(0.1)
    since = time.time()
    time.sleep(0.3)
    assert estop_events(run.node) == []
    assert run.outputs(since=since) and all(
        carries(msg, run, shoulder=moved) for _, msg in run.outputs(since=since))
    assert f'new writer on {UPPER} accepted' in run.node.output()


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_a_second_writer_on_the_lower_input_before_engaging(stack, robot):
    """SAFETY: a second writer latches even before the other input engages."""
    run = stack(robot, 'split' if robot == 'g1' else 'sim_safety_split')
    joints = len(run.robot.profile.joints)
    (lower_pub, lower), (upper_pub, upper) = input_pairs(run)
    second = run.robot.publisher(LOWER)
    # The lower input alone first, which engages on its first command.
    run.robot.publish_for(0.2, (lower_pub, lower), (second, lower))
    run.robot.publish_for(0.3, (lower_pub, lower), (upper_pub, upper))
    time.sleep(0.2)
    # The upper input never engaged; then the estop command: the H1-2's burst and then
    # silence, the G1's every cycle.
    assert all(not carries(msg, run) for _, msg in run.outputs())
    assert is_zeroed(run.outputs()[-1][1], joints)
    if robot == 'g1':
        assert run.outputs()[-1][0] > time.time() - 0.1, 'keep_writing'
    else:
        assert run.outputs()[-1][0] < time.time() - 0.1
    (reason,) = reasons(run.node)
    assert reason.startswith(f'second writer on {LOWER}: '), reason


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_heartbeat_true_then_false_on_estop(stack, robot):
    """The heartbeat is true at 10 Hz while relaying and false at 10 Hz once latched."""
    run = stack(robot, SPLIT[robot])
    pairs = input_pairs(run)
    run.robot.stream(*pairs)
    time.sleep(1.2)
    now = time.time()
    beats = run.robot.outputs(HEARTBEAT)
    assert beats and all(beat for _, beat in beats)
    assert 8 <= len([t for t, _ in beats if now - 1.0 <= t < now]) <= 12, '10 Hz'

    send_once(run.robot, pairs[0][0], run.robot.command(knee=float('nan')), pairs)
    latched = time.time()
    time.sleep(1.2)
    after = run.robot.outputs(HEARTBEAT, since=latched + 0.15)
    assert after and not any(beat for _, beat in after)
    now = time.time()
    assert 8 <= len([t for t, _ in after if now - 1.0 <= t < now]) <= 12, '10 Hz'


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_no_heartbeat_while_lowstate_is_another_robots(stack, robot):
    """No heartbeat while /lowstate is another robot's; it starts, true, on the robot's own."""
    run = stack(robot, SPLIT[robot], stream_state=False)
    own = run.robot.profile.mode_machine
    run.robot.state.mode_machine = 5 if own == 6 else 6
    run.robot.resume()
    run.robot.stream(*input_pairs(run))
    time.sleep(1.0)
    assert run.robot.outputs(HEARTBEAT) == [], 'the relay cannot begin on a foreign /lowstate'
    assert estop_events(run.node) == []

    switched = time.time()
    run.robot.state.mode_machine = own
    time.sleep(1.0)
    beats = run.robot.outputs(HEARTBEAT)
    assert beats and all(beat for _, beat in beats)
    assert beats[0][0] > switched
    assert run.outputs() and carries(run.outputs()[-1][1], run)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
@pytest.mark.parametrize('latch', ['command', 'estop'])
def test_heartbeat_false_after_a_latch_before_the_first_beat(stack, robot, latch):
    """
    SAFETY: a latch before the relay was ready makes the beats false and never true.

    command: a non-finite command with no /lowstate yet; estop: a first hardware estop sample
    that says triggered, with the robot's /lowstate.
    """
    if latch == 'command':
        run = stack(robot, SPLIT[robot], stream_state=False)
        pub = run.robot.publisher(UPPER)
        pub.publish(run.robot.command(shoulder=float('nan')))
        reason_prefix = 'Command validation failed on /safety/lowcmd_upper_in'
    else:
        run = stack(robot, SPLIT[robot], use_estop=True)
        time.sleep(0.3)
        run.robot.estop_values = (True, True)
        reason_prefix = 'External estop triggered'
    time.sleep(1.2)
    (report,) = run.node.events('estop_command_written')
    assert report['reason'].startswith(reason_prefix), report
    beats = run.robot.outputs(HEARTBEAT)
    assert beats and not any(beat for _, beat in beats)
    now = time.time()
    assert 8 <= len([t for t, _ in beats if now - 1.0 <= t < now]) <= 12, '10 Hz'


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_estop_topics_gate_engagement(stack, robot):
    """
    SAFETY: with use_estop, nothing engages and no heartbeat goes out until a sample has a
    value on both estop topics.
    """
    run = stack(robot, SPLIT[robot], use_estop=True)
    joints = len(run.robot.profile.joints)
    run.robot.stream(*input_pairs(run))
    time.sleep(0.5)

    def not_engaged() -> bool:
        """Return whether every published command so far is the zero command."""
        published = run.outputs()
        return bool(published) and all(is_zeroed(msg, joints) for _, msg in published)

    assert not_engaged()
    assert reasons(run.node, 'engage_refused')[0] == (
        'no hardware estop sample received yet on /estop/triggered or /estop/plugged_in')
    # A sample is a value on each topic.
    run.robot.estop_values = (False, None)
    time.sleep(0.5)
    assert not_engaged()
    # No heartbeat before the relay can begin.
    assert run.robot.outputs(HEARTBEAT) == []
    sampled = time.time()
    run.robot.estop_values = (False, True)
    time.sleep(0.5)
    assert run.outputs() and carries(run.outputs()[-1][1], run)
    assert estop_events(run.node) == []
    beats = run.robot.outputs(HEARTBEAT)
    assert beats and all(beat for _, beat in beats)
    assert beats[0][0] > sampled


@pytest.mark.parametrize('robot', ['h12', 'g1'])
@pytest.mark.parametrize('values, reason', [
    ((True, True), 'External estop triggered'), ((False, False), 'External estop unplugged'),
], ids=['triggered', 'unplugged'])
def test_estop_first_sample_latches(stack, robot, values, reason):
    """SAFETY: a first hardware estop sample that is triggered or unplugged latches."""
    run = stack(robot, SPLIT[robot], use_estop=True)
    run.robot.stream(*input_pairs(run))
    run.robot.estop_values = values
    time.sleep(0.5)
    published = run.outputs()
    joints = len(run.robot.profile.joints)
    assert published and all(is_zeroed(msg, joints) for _, msg in published)
    if robot == 'g1':
        assert published[-1][0] > time.time() - 0.1, 'keep_writing'
    else:
        assert published[-1][0] < time.time() - 0.2, 'the burst, then silence'
    (report,) = run.node.events('estop_command_written')
    assert 'trigger_to_action_ms' in report
    assert report['reason'] == reason and report['trigger_stamp_ms'] is not None
    assert run.robot.outputs(HEARTBEAT)[-1][1] is False


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_estop_silent_after_engaging_latches(stack, robot):
    """SAFETY: a hardware estop silent for its timeout latches."""
    run = stack(robot, SPLIT[robot], use_estop=True)
    run.robot.estop_values = (False, True)
    run.robot.stream(*input_pairs(run))
    time.sleep(0.5)
    assert run.outputs() and carries(run.outputs()[-1][1], run)
    run.robot.estop_values = None  # the estop node died
    time.sleep(0.8)
    (reason,) = reasons(run.node)
    prefix = 'External estop silent: no sample on /estop/triggered or /estop/plugged_in for '
    assert reason.startswith(prefix) and reason.endswith(' s (timeout 0.5 s)'), reason
    assert 0.5 <= float(reason[len(prefix):].split(' ')[0]) < 0.6  # to 3 decimals
    assert is_zeroed(run.outputs()[-1][1], len(run.robot.profile.joints))


def test_estop_topics_are_ignored_without_use_estop(stack):
    """Without use_estop the estop topics are ignored."""
    run = stack('h12', 'sim_safety_split')
    run.robot.estop_values = (True, False)
    run.robot.stream(*input_pairs(run))
    time.sleep(0.5)
    assert run.outputs() and carries(run.outputs()[-1][1], run)
    assert estop_events(run.node) == []
    assert all(beat for _, beat in run.robot.outputs(HEARTBEAT))


EXIT_LOWCMD = [('h12', 'sim_safety_split'), ('h12', 'default_safety_full'), ('g1', 'split'),
               ('g1', 'full')]


@pytest.mark.parametrize('robot, config', EXIT_LOWCMD, ids=[f'{r}-{c}' for r, c in EXIT_LOWCMD])
def test_sigint_split_stops_publishing(stack, robot, config):
    """SAFETY: SIGINT stops publishing at once without an estop, exiting 0."""
    run = stack(robot, config)
    # The controllers keep commanding through the exit.
    run.robot.stream(*input_pairs(run))
    run.robot.wait_for_output('/lowcmd', lambda msg: carries(msg, run))
    time.sleep(0.2)
    sent = time.time()
    run.node.proc.send_signal(signal.SIGINT)
    code = run.node.wait_exit(5.0)
    exited = time.time()
    time.sleep(0.2)
    assert code == 0, run.node.output()
    assert run.outputs(since=sent + 0.05) == []
    before = run.outputs()
    assert carries(before[-1][1], run), 'the robot holds the last relayed command'
    assert exited - sent < 2.0
    assert run.node.events('exit_requested') == [
        {'event': 'exit_requested', 'reason': 'Exit requested (SIGINT)'}]
    assert estop_events(run.node) == []


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_an_exit_before_engaging_stops_the_zero_stream_at_once(stack, robot):
    """SIGTERM before engaging stops the zero stream at once, exiting 0."""
    run = stack(robot, SPLIT[robot])
    joints = len(run.robot.profile.joints)
    time.sleep(0.2)
    sent = time.time()
    run.node.proc.send_signal(signal.SIGTERM)
    code = run.node.wait_exit(5.0)
    elapsed = time.time() - sent
    time.sleep(0.1)
    assert code == 0, run.node.output()
    assert elapsed < 1.5
    published = run.outputs()
    assert published and all(is_zeroed(msg, joints) for _, msg in published)
    assert run.outputs(since=sent + 0.05) == []
    assert run.node.events('exit_requested') == [
        {'event': 'exit_requested', 'reason': 'Exit requested (SIGTERM)'}]
    assert estop_events(run.node) == []


def _stdout_pipe(stalled: bool) -> tuple[int, int]:
    """Return a pipe (read end, write end); stalled: already full, so a write blocks."""
    import fcntl
    read_fd, write_fd = os.pipe()
    if stalled:
        # Nobody reads it: fill it (shrunk to one page), so that the node's first line blocks
        # its writer for good.
        fcntl.fcntl(write_fd, fcntl.F_SETPIPE_SZ, 4096)
        os.set_blocking(write_fd, False)
        try:
            while True:
                os.write(write_fd, b'x' * 4096)
        except BlockingIOError:
            pass
        # The node shares this open file: blocking again, as ros2 launch's pipe is.
        os.set_blocking(write_fd, True)
    return read_fd, write_fd


@pytest.mark.parametrize('stdout, trigger', [
    ('closed', 'estop'), ('closed', signal.SIGTERM), ('closed', signal.SIGHUP),
    ('stalled', 'estop'), ('stalled', signal.SIGTERM),
], ids=['closed-estop', 'closed-SIGTERM', 'closed-SIGHUP', 'stalled-estop', 'stalled-SIGTERM'])
def test_a_closed_or_stalled_stdout_never_stops_the_estop_or_the_exit(tmp_path, fake_robot,
                                                                     stdout, trigger):
    """SAFETY: a closed or stalled stdout stops neither the estop command nor a clean exit."""
    # Logging on, as in the shipped config: stopping the chunk logger runs in the exit too.
    path, raw = write_config(tmp_path, 'g1', 'sim_split', logging={
        'enabled': True, 'base_dir': str(tmp_path / 'logs')})
    robot = fake_robot('g1')
    joints = len(robot.profile.joints)
    shoulder = robot.profile.shoulder
    read_fd, write_fd = _stdout_pipe(stalled=stdout == 'stalled')
    # stdout and stderr together: ros2 launch killed, a closed terminal, or a reader that
    # stopped reading.
    node = SafetyNode('g1', path, stdout=write_fd, stderr=write_fd)
    os.close(write_fd)  # only the node holds the write end now
    try:
        lower, upper = robot.publisher(LOWER), robot.publisher(UPPER)
        pairs = [(lower, robot.command(knee=KNEE_Q)), (upper, robot.command(shoulder=SHOULDER_Q))]
        robot.stream(*pairs)
        robot.wait_for_output('/lowcmd', lambda msg: close(msg.motor_cmd[shoulder].q, SHOULDER_Q),
                              timeout=30.0)
        if stdout == 'closed':
            os.close(read_fd)  # the node has written its start-up lines; nobody reads now
            read_fd = None
        if trigger == 'estop':
            # An invalid command: the estop's lines are written as its first command goes out,
            # and keep_writing writes that command every cycle.
            send_once(robot, upper, robot.command(shoulder=float('nan')), pairs)
            robot.wait_for_output('/lowcmd', lambda msg: is_zeroed(msg, joints), timeout=2.0)
            latched = time.time()
            time.sleep(0.5)
            written = robot.outputs('/lowcmd', since=latched)
            assert len(written) > 100 and all(is_zeroed(msg, joints) for _, msg in written)
            signalled = time.time()
            node.proc.send_signal(signal.SIGTERM)
        else:
            signalled = time.time()
            node.proc.send_signal(trigger)
        code = node.wait_exit(8.0)
        exited = time.time()
    finally:
        node.stop()
        if read_fd is not None:
            os.close(read_fd)

    # split_mode stops publishing at once: a clean exit, delayed by a stalled stdout for at
    # most one 1 s wait for the last lines.
    assert code == 0
    assert robot.outputs('/lowcmd', since=signalled + 0.05) == []
    assert exited - signalled < 3.5
    assert list((tmp_path / 'logs').glob('*/chunk_*.npz')), 'the logger must have written'
