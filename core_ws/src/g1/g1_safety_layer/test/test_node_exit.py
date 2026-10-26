'''
SIGINT/SIGTERM/SIGHUP on the safety node and script as subprocesses, also with
stdout closed or stalled (domain 97, no robot)
'''

import os
import sys
import json
import time
import fcntl
import signal
import itertools
import threading
import subprocess
from pathlib import Path

import yaml
import pytest

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from g1_safety_layer.core.joint_limits import ARM_SDK_WEIGHT_SLOT, MOTOR_COUNT

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / 'config'
SCRIPT = PACKAGE_DIR / 'g1_safety_layer' / 'script' / 'safety_layer_main.py'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
KNEE = 3
SHOULDER = 15
KNEE_Q = 0.5
SHOULDER_Q = 0.2
_factory = []
_runs = itertools.count()


def _init_dds():
    # the Unitree channel factory is process-wide; initialize it once
    if not _factory:
        ChannelFactoryInitialize(TEST_DOMAIN)
        _factory.append(True)


def _write_config(tmp_path: Path, name: str, **changes) -> tuple[Path, dict]:
    '''
    A copy of config/name on topics of its own and domain 97, without logging;
    each keyword argument replaces a setting, or updates a section with a dict
    '''
    raw = yaml.safe_load((CONFIG_DIR / name).read_text())
    prefix = f'test_node_exit/{os.getpid()}/{next(_runs)}/'
    raw['topics'] = {key: prefix + topic for key, topic in raw['topics'].items()}
    # ROS_DOMAIN_ID below says 97 too: neither source may be domain 0
    raw['network']['domain_id'] = TEST_DOMAIN
    raw['logging']['enabled'] = False
    for key, value in changes.items():
        if isinstance(value, dict):
            raw[key].update(value)
        else:
            raw[key] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path, raw


def _command_line(entry: str, config_path: Path) -> list[str]:
    '''The installed safety_node (what ros2 launch runs), the module, or the script'''
    if entry == 'script':
        return [sys.executable, str(SCRIPT), '--config', str(config_path)]
    pytest.importorskip('rclpy')
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = Path(get_package_prefix('g1_safety_layer'))
        node = prefix / 'lib' / 'g1_safety_layer' / 'safety_node'
        if node.exists():
            return [str(node), '--config', str(config_path)]
    except Exception:
        pass
    return [sys.executable, '-m', 'g1_safety_layer.ros2.safety_node', '--config', str(config_path)]


def _env(unbuffered: bool = True, log_dir: Path | None = None) -> dict:
    '''The environment of the node: domain 97, no interface pin, logs only to log_dir'''
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(TEST_DOMAIN)
    env.pop('PYTHONUNBUFFERED', None)
    if unbuffered:
        env['PYTHONUNBUFFERED'] = '1'
    env.pop('G1_DDS_INTERFACE', None)
    env.pop('G1_SAFETY_LOG_DIR', None)
    if log_dir is not None:
        env['G1_SAFETY_LOG_DIR'] = str(log_dir)
    return env


class _Process:
    '''The node as a subprocess, with its output lines collected by a thread'''
    def __init__(self, command: list[str]) -> None:
        self.lines: list[str] = []
        self.proc = subprocess.Popen(
            command, cwd=PACKAGE_DIR, env=_env(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.lines if line.startswith('{')]

    def wait_for_event(self, name: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(e['event'] == name for e in self.events()):
                return
            assert self.proc.poll() is None, '\n'.join(self.lines)
            time.sleep(0.05)
        raise AssertionError(f'no {name} within {timeout} s:\n' + '\n'.join(self.lines))

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self._reader.join(timeout=2.0)


def _state() -> LowState_:
    state = LowStateDefault()
    state.mode_machine = 5
    state.motor_state[KNEE].q = KNEE_Q
    state.motor_state[SHOULDER].q = SHOULDER_Q
    return state


def _command(knee: float | None = None, shoulder: float | None = None) -> LowCmd_:
    '''Like the controller's: the waist held at 0 with kp 60, and the given joints'''
    cmd = LowCmdDefault()
    for i in (12, 13, 14):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].kp = 60.0
        cmd.motor_cmd[i].kd = 1.5
    for slot, q in ((KNEE, knee), (SHOULDER, shoulder)):
        if q is not None:
            cmd.motor_cmd[slot].mode = 1
            cmd.motor_cmd[slot].q = q
            cmd.motor_cmd[slot].kp = 60.0
            cmd.motor_cmd[slot].kd = 1.5
    return cmd


class _Robot:
    '''rt/lowstate and the controller's commands, streamed until close()'''
    def __init__(self, topics: dict, inputs: list[tuple[str, LowCmd_]]) -> None:
        _init_dds()
        self.received = []
        # written once in place of the first input's next command
        self.once: LowCmd_ | None = None
        self._sub = ChannelSubscriber(topics['low_cmd_out'], LowCmd_)
        self._sub.Init(lambda m: self.received.append((time.time(), m)), 50)
        self._state_pub = ChannelPublisher(topics['low_state'], LowState_)
        self._state_pub.Init()
        self._inputs = []
        for key, cmd in inputs:
            pub = ChannelPublisher(topics[key], LowCmd_)
            pub.Init()
            self._inputs.append((pub, cmd))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        state = _state()
        tick = 0
        while not self._stop.is_set():
            self._state_pub.Write(state)
            if tick % 2 == 0:  # rt/lowstate at 200 Hz, commands at 100 Hz
                for pub, cmd in self._inputs:
                    once, self.once = self.once, None
                    pub.Write(cmd if once is None else once)
            tick += 1
            time.sleep(0.005)

    def wait_for(self, check, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.received and check(self.received[-1][1]):
                return
            time.sleep(0.02)
        raise AssertionError('the layer never published the expected command')

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sub.Close()
        self._state_pub.Close()
        for pub, _ in self._inputs:
            pub.Close()


@pytest.mark.parametrize('entry, signum', [
    ('node', signal.SIGTERM),
    ('node', signal.SIGINT),
    ('node', signal.SIGHUP),
    ('script', signal.SIGTERM),
], ids=['node-SIGTERM', 'node-SIGINT', 'node-SIGHUP', 'script-SIGTERM'])
def test_an_exit_signal_releases_the_arms_then_exits(tmp_path, entry, signum):
    config_path, raw = _write_config(tmp_path, 'arm_sdk.yaml')
    release = raw['arm_sdk']['release_seconds']  # the shipped 3 s
    process = _Process(_command_line(entry, config_path))
    robot = None
    try:
        process.wait_for_event('relay_started', timeout=30.0)
        # the controller keeps commanding through the exit, the worst case
        robot = _Robot(raw['topics'], [('low_cmd_upper_in', _command(shoulder=SHOULDER_Q))])
        robot.wait_for(lambda m: m.motor_cmd[ARM_SDK_WEIGHT_SLOT].q == 1.0, timeout=10.0)
        sent = time.time()
        process.proc.send_signal(signum)
        code = process.proc.wait(timeout=release + 5.0)
        exited = time.time()
    finally:
        if robot is not None:
            robot.close()
        process.close()

    output = '\n'.join(process.lines)
    assert code == 0, output
    after = [(t, m.motor_cmd[ARM_SDK_WEIGHT_SLOT].q, m.motor_cmd[SHOULDER].q)
             for t, m in robot.received if t >= sent]
    weights = [w for _, w, _ in after]
    assert weights[0] == 1.0 and weights[-1] == 0.0, output
    assert all(b <= a for a, b in zip(weights, weights[1:])), 'the release must be monotonic'
    assert all(abs(q - SHOULDER_Q) < 1e-6 for _, _, q in after), 'must hold the last good target'
    ramp_start = next(t for t, w, _ in after if w < 1.0)
    ramp_end = next(t for t, w, _ in after if w == 0.0)
    assert release - 0.15 < ramp_end - ramp_start < release + 0.15
    assert exited - sent < release + 2.0
    exits = [e for e in process.events() if e['event'] == 'exit_requested']
    assert exits == [{'event': 'exit_requested', 'engaged': True,
                      'reason': f'Exit requested ({signal.Signals(signum).name})'}]


def test_an_exit_signal_in_split_mode_writes_the_estop_command_then_exits(tmp_path):
    config_path, raw = _write_config(tmp_path, 'split.yaml')
    process = _Process(_command_line('node', config_path))
    robot = None
    try:
        process.wait_for_event('relay_started', timeout=30.0)
        robot = _Robot(raw['topics'], [('low_cmd_lower_in', _command(knee=KNEE_Q)),
                                       ('low_cmd_upper_in', _command(shoulder=SHOULDER_Q))])
        robot.wait_for(lambda m: abs(m.motor_cmd[KNEE].q - KNEE_Q) < 1e-6, timeout=10.0)
        sent = time.time()
        process.proc.send_signal(signal.SIGTERM)
        code = process.proc.wait(timeout=5.0)
        exited = time.time()
    finally:
        if robot is not None:
            robot.close()
        process.close()

    output = '\n'.join(process.lines)
    assert code == 0, output
    after = [(t, m) for t, m in robot.received if t >= sent]

    def zeroed(msg: LowCmd_) -> bool:
        return all(m.mode == 0 and m.q == m.kp == m.kd == m.tau == 0.0
                   for m in msg.motor_cmd[:MOTOR_COUNT])

    idx = next(i for i, (_, m) in enumerate(after) if zeroed(m))
    assert all(zeroed(m) for _, m in after[idx:]), 'the estop command must be the last one'
    assert after[-1][0] - after[idx][0] > 0.08
    assert exited - sent < 2.0
    reasons = [e['reason'] for e in process.events() if e['event'] == 'estop_command_written']
    assert reasons == ['Exit requested (SIGTERM)']


def test_the_node_exits_non_zero_when_the_layer_fails_to_start(tmp_path):
    config_path, _ = _write_config(tmp_path, 'arm_sdk.yaml', mode='no_such_mode')
    process = _Process(_command_line('node', config_path))
    try:
        code = process.proc.wait(timeout=30.0)
    finally:
        process.close()
    output = '\n'.join(process.lines)
    assert code == 1, output
    assert 'Safety layer exited with error: Unknown mode: no_such_mode' in output


def _stdout_pipe(stalled: bool) -> tuple[int, int]:
    '''A pipe (read end, write end); stalled: already full, so that a write blocks'''
    read_fd, write_fd = os.pipe()
    if stalled:
        # nobody reads it: fill it (shrunk to one page), then the node's first
        # line blocks its writer for good
        fcntl.fcntl(write_fd, fcntl.F_SETPIPE_SZ, 4096)
        os.set_blocking(write_fd, False)
        try:
            while True:
                os.write(write_fd, b'x' * 4096)
        except BlockingIOError:
            pass
        # the node shares this open file: blocking again, as ros2 launch's pipe is
        os.set_blocking(write_fd, True)
    return read_fd, write_fd


@pytest.mark.parametrize('entry, stdout, trigger', [
    ('node', 'closed', 'estop'),
    ('node', 'closed', signal.SIGTERM),
    ('node', 'closed', signal.SIGHUP),
    ('node', 'stalled', 'estop'),
    ('node', 'stalled', signal.SIGTERM),
    ('script', 'closed', signal.SIGTERM),
], ids=['node-closed-estop', 'node-closed-SIGTERM', 'node-closed-SIGHUP',
        'node-stalled-estop', 'node-stalled-SIGTERM', 'script-closed-SIGTERM'])
def test_a_closed_or_stalled_stdout_never_stops_the_hand_back(tmp_path, entry, stdout, trigger):
    release = 1.0
    # logging on, as in the shipped config: stopping its logger writes a line too
    config_path, raw = _write_config(tmp_path, 'arm_sdk.yaml',
                                     arm_sdk={'release_seconds': release},
                                     logging={'enabled': True})
    command = _command_line(entry, config_path)
    read_fd, write_fd = _stdout_pipe(stalled=stdout == 'stalled')
    # closed: stdout and stderr go together (ros2 launch killed, a closed
    # terminal); stalled: stderr to a file, since rclpy logs there at startup
    stderr = write_fd if stdout == 'closed' else (tmp_path / 'stderr.txt').open('w')
    # block-buffered stdout, as under ros2 launch
    env = _env(unbuffered=False, log_dir=tmp_path / 'logs')
    proc = subprocess.Popen(command, cwd=PACKAGE_DIR, env=env, stdout=write_fd, stderr=stderr)
    # only the node holds the write end now
    os.close(write_fd)
    if stderr is not write_fd:
        stderr.close()
    robot = None
    try:
        robot = _Robot(raw['topics'], [('low_cmd_upper_in', _command(shoulder=SHOULDER_Q))])
        robot.wait_for(lambda m: m.motor_cmd[ARM_SDK_WEIGHT_SLOT].q == 1.0, timeout=30.0)
        if stdout == 'closed':
            # the node has written its startup lines; nobody reads any more
            os.close(read_fd)
            read_fd = None
        sent = time.time()
        if trigger == 'estop':
            # an invalid command: the estop's lines are written as the ramp starts
            robot.once = _command(shoulder=float('nan'))
            robot.wait_for(lambda m: m.motor_cmd[ARM_SDK_WEIGHT_SLOT].q == 0.0,
                           timeout=release + 2.0)
            time.sleep(0.2)
            proc.send_signal(signal.SIGTERM)
        else:
            proc.send_signal(trigger)
        code = proc.wait(timeout=release + 6.0)
        exited = time.time()
    finally:
        if robot is not None:
            robot.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if read_fd is not None:
            os.close(read_fd)

    after = [(t, m.motor_cmd[ARM_SDK_WEIGHT_SLOT].q, m.motor_cmd[SHOULDER].q)
             for t, m in robot.received if t >= sent]
    weights = [w for _, w, _ in after]
    assert weights and weights[0] == 1.0 and weights[-1] == 0.0, weights[-5:]
    assert all(b <= a for a, b in zip(weights, weights[1:])), 'the release must be monotonic'
    assert all(abs(q - SHOULDER_Q) < 1e-6 for _, _, q in after), 'must hold the last good target'
    ramp_start = next(t for t, w, _ in after if w < 1.0)
    ramp_end = next(t for t, w, _ in after if w == 0.0)
    assert release - 0.15 < ramp_end - ramp_start < release + 0.15
    # a clean exit, delayed by a stalled stdout for at most two 1 s waits for
    # the last lines
    assert code == 0
    assert exited - ramp_end < 3.5
    assert list((tmp_path / 'logs').glob('*/chunk_*.npz')), 'the logger must have written'
