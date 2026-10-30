#!/usr/bin/env python3
"""Measure frame_task_server's command rate with its IK running, from another process.

Starts the installed frame_task_server for --robot with --config as a subprocess. This process
plays the robot and golem_safety's relay: it publishes /lowstate at 500 Hz, executes every
command on /safety/lowcmd_upper_in at once (each slot with kp > 0 jumps to its q) and relays
it on the config's topics.safety_out, and records the receive time of every command. Two
windows are measured: the start-up (from the first command to the server's first
/left_ee_pose, the start-up moves stepping the IK) and the goals (FrameTask goals that move
both end effectors 10 cm forward and back, one after another, for --seconds). Then the
server is stopped with SIGINT.

The JSON on stdout gives, per window, the achieved rate, the 99th percentile of the receive
period, the longest gap, how many gaps were over 20 ms and the control steps per second
(distinct commands), in the units named by the keys. Then the server's threads once it is up:
how many there are, and how many carry the process's own name (Python's threads and native
threads that do not name themselves, such as OpenBLAS's workers; DDS names its own); then the
server's exit code and whether its last command was mode 0 with zero gains. The server's
output goes to stderr if it fails.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

# OpenBLAS starts one busy-waiting worker per core by default, whose spin adds jitter to this
# process's 500 Hz /lowstate thread. Set before numpy loads BLAS.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

from ament_index_python.packages import get_package_prefix  # noqa: E402
from custom_ros_messages.action import FrameTask  # noqa: E402
from geometry_msgs.msg import Pose, PoseStamped  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from unitree_hg.msg import LowCmd, LowState  # noqa: E402

from golem_upper_ik.config import load_controller_config  # noqa: E402
from golem_upper_ik.profile import load_profile  # noqa: E402

READY_TIMEOUT_S = 120.0
LONG_GAP_MS = 20.0
GOAL_TIMEOUT_S = 30.0
RECORD_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST, depth=10000)
WRITER_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)


class Robot:
    """/lowstate at 500 Hz; executes, relays and records every command"""

    def __init__(self, node, profile, config):
        self.profile = profile
        count = len(profile.body_joints)
        self._lock = threading.Lock()
        self._q = np.zeros(count)
        self._upper = list(profile.upper_slots)
        self.times = []
        self.upper_q = []
        self.last = None
        self._relay = None
        self._state = LowState()
        self._state.mode_machine = profile.mode_machine
        self._state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        self._state_publisher = node.create_publisher(LowState, config['topics']['low_state'],
                                                      WRITER_QOS)
        self._relay_publisher = node.create_publisher(LowCmd, config['topics']['safety_out'],
                                                      WRITER_QOS)
        node.create_subscription(LowCmd, config['topics']['low_cmd'], self._on_command,
                                 RECORD_QOS)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _on_command(self, msg):
        now = time.monotonic()
        motors = msg.motor_cmd
        q = tuple(motors[slot].q for slot in self._upper)
        with self._lock:
            self.times.append(now)
            self.upper_q.append(q)
            for slot in range(len(self._q)):
                if motors[slot].kp > 0.0:
                    self._q[slot] = motors[slot].q
            self._relay = msg
            self.last = msg

    def _run(self):
        start = time.monotonic()
        next_at = start
        while not self._stop.is_set():
            with self._lock:
                q = self._q.copy()
                relay = self._relay
            self._state.tick = int((time.monotonic() - start) * 1000.0)
            for slot, value in enumerate(q):
                self._state.motor_state[slot].q = float(value)
            self._state_publisher.publish(self._state)
            if relay is not None:
                self._relay_publisher.publish(relay)
            next_at += 0.002
            time.sleep(max(0.0, next_at - time.monotonic()))

    def window(self, start, end):
        """(receive times, upper q) of the commands received from start to end"""
        with self._lock:
            pairs = [(t, q) for t, q in zip(self.times, self.upper_q) if start <= t <= end]
        return [t for t, _ in pairs], [q for _, q in pairs]

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


def stats(times, upper_q):
    times = np.asarray(times)
    if len(times) < 2:
        return {'received': len(times)}
    gaps = np.diff(times) * 1000.0
    span = times[-1] - times[0]
    steps = 1 + sum(1 for a, b in zip(upper_q, upper_q[1:]) if a != b)
    return {
        'received': len(times),
        'seconds': round(float(span), 3),
        'mean_hz': round((len(times) - 1) / span, 3),
        'p99_period_ms': round(float(np.percentile(gaps, 99)), 3),
        'max_gap_ms': round(float(gaps.max()), 3),
        'gaps_over_20ms': int(np.sum(gaps > LONG_GAP_MS)),
        'control_steps_hz': round(steps / span, 3),
    }


def threads_of(pid):
    '''(threads, threads named as the process) of the process pid, from /proc'''
    task = f'/proc/{pid}/task'
    with open(f'/proc/{pid}/comm', encoding='utf-8') as handle:
        name = handle.read().strip()
    names = []
    for tid in os.listdir(task):
        try:
            with open(f'{task}/{tid}/comm', encoding='utf-8') as handle:
                names.append(handle.read().strip())
        except FileNotFoundError:  # the thread ended
            pass
    return len(names), sum(1 for thread in names if thread == name)


def wait(condition, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return bool(condition())


def forward(x):
    pose = Pose()
    pose.position.x = x
    pose.orientation.w = 1.0
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--robot', default='h12')
    parser.add_argument('--config', default='safety_split')
    parser.add_argument('--seconds', type=float, default=6.0)
    args = parser.parse_args()

    profile = load_profile(args.robot)
    config = load_controller_config(args.robot, args.config)
    rclpy.init()
    node = rclpy.create_node('frame_task_server_rate_probe')
    robot = Robot(node, profile, config)
    ready = []
    node.create_subscription(PoseStamped, '/left_ee_pose',
                             lambda msg: ready or ready.append(time.monotonic()), 10)
    client = ActionClient(node, FrameTask, '/frame_task')
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # the server starts without the BLAS thread settings of this process's environment, so
    # its thread count shows the node's own pin
    server_env = {key: value for key, value in os.environ.items()
                  if key not in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}
    server = subprocess.Popen(
        [str(os.path.join(get_package_prefix('golem_upper_ik'), 'lib', 'golem_upper_ik',
                          'frame_task_server')),
         '--ros-args', '-p', f'robot:={args.robot}', '-p', f'config:={args.config}'],
        env=server_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    threading.Thread(target=lambda: output.extend(server.stdout), daemon=True).start()
    result = {'robot': args.robot, 'config': args.config}
    code = 0
    try:
        if not wait(lambda: ready, READY_TIMEOUT_S):
            raise RuntimeError('frame_task_server did not publish /left_ee_pose')
        result['startup'] = stats(*robot.window(robot.times[0], ready[0]))
        result['server_threads'], result['server_threads_named_as_process'] = \
            threads_of(server.pid)

        assert client.wait_for_server(timeout_sec=10.0), 'no /frame_task action server'
        goals_start = time.monotonic()
        sent = 0
        while time.monotonic() - goals_start < args.seconds:
            goal = FrameTask.Goal()
            goal.frame_names = list(profile.ee_frames)
            x = 0.1 if sent % 2 == 0 else -0.1
            goal.frame_targets = [forward(x), forward(x)]
            goal.relative = True
            future = client.send_goal_async(goal)
            if not wait(future.done, 10.0) or not future.result().accepted:
                raise RuntimeError('a FrameTask goal was not accepted')
            done = future.result().get_result_async()
            if not wait(done.done, GOAL_TIMEOUT_S):
                raise RuntimeError('a FrameTask goal gave no result')
            sent += 1
        goals_end = time.monotonic()
        result['goals'] = stats(*robot.window(goals_start, goals_end))
        result['goals_sent'] = sent
    except Exception as error:
        result['error'] = repr(error)
        code = 1
    finally:
        server.send_signal(signal.SIGINT)
        try:
            result['server_exit_code'] = server.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            server.kill()
            result['server_exit_code'] = server.wait(timeout=5.0)
        time.sleep(0.3)
        last = robot.last
        result['shutdown_zero_gain'] = last is not None and all(
            last.motor_cmd[slot].mode == 0 and last.motor_cmd[slot].kp == 0.0 and
            last.motor_cmd[slot].kd == 0.0 for slot in profile.upper_slots)
        robot.stop()
        executor.shutdown(timeout_sec=2.0)
        rclpy.try_shutdown()
    if code or result['server_exit_code'] != 0:
        print(''.join(output), file=sys.stderr)
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    # The executor's thread pool may still start a callback it was handed before its
    # shutdown() returned, so nothing is destroyed: the process ends here, as the nodes' do
    # (golem_upper_ik.ros2.utility.run)
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
