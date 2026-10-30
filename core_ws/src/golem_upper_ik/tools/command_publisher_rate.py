#!/usr/bin/env python3
"""Measure CommandPublisher's publish rate beside IK-sized CPU load.

One process. A CommandPublisher (golem_upper_ik.core.ros_io) sends the H1-2 profile's
commands on /safety/rate_probe_upper_in at --rate. The main thread runs 30 Hz "IK steps"
that burn --burn-ms of numpy work and then send a new command, as the controller does. A
listener node subscribes with a QoS that keeps every sample and records each receive time.
A multi-threaded executor thread spins the nodes.

With --gate the commands go through a LowCmdHandler whose follow gate stays shut (nothing
relays them), so its check_hz (1000 Hz) gate thread runs throughout. It reads the measured
pose from a RobotModel whose StateSubscriber takes an in-process /lowstate stream at 500 Hz,
and each step commands a pose 0.3 rad away, which the gate holds near the measured one.

The JSON on stdout is the receive side: the achieved rate, the 99th percentile of the
receive period and the longest gap, in the units named by the keys, and with --gate whether
the gate stayed shut. Counts go to stderr.
"""
import argparse
import json
import os
import sys
import threading
import time

# OpenBLAS starts one busy-waiting worker per core by default, and their spin adds several
# ms to tens of ms of jitter to the publisher thread. Set before numpy loads BLAS.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from unitree_hg.msg import LowCmd, LowState  # noqa: E402

from golem_upper_ik.core.ros_io import CommandPublisher  # noqa: E402
from golem_upper_ik.profile import load_profile  # noqa: E402

TOPIC = '/safety/rate_probe_upper_in'
DISCOVERY_TIMEOUT_S = 10.0
SPARE_SAMPLES = 1000


class Recorder:
    """Subscribes with reliable, deep keep-last QoS and stamps every receive."""

    def __init__(self, node, capacity):
        self.times = np.zeros(capacity)
        self.count = 0
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10000)
        self.subscription = node.create_subscription(LowCmd, TOPIC, self._on_msg, qos)

    def _on_msg(self, _msg):
        if self.count < len(self.times):
            self.times[self.count] = time.monotonic()
        self.count += 1


class StateStream:
    """/lowstate at rate_hz from a thread: one prebuilt message, its tick advancing."""

    def __init__(self, node, profile, q, rate_hz):
        self._publisher = node.create_publisher(LowState, '/lowstate', 10)
        self._msg = LowState()
        self._msg.mode_machine = profile.mode_machine
        self._msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        for slot, value in enumerate(q):
            self._msg.motor_state[slot].q = float(value)
        self._period = 1.0 / rate_hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        start = time.monotonic()
        next_at = start
        while not self._stop.is_set():
            self._msg.tick = int((time.monotonic() - start) * 1000.0)
            self._publisher.publish(self._msg)
            next_at += self._period
            time.sleep(max(0.0, next_at - time.monotonic()))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


def publisher_sender(controller, profile, rate):
    """send(step) through a bare CommandPublisher, and its stop()"""
    publisher = CommandPublisher(controller, TOPIC, rate, profile)
    count = len(profile.upper_slots)
    zeros = np.zeros(count)
    gains = np.full(count, 100.0)

    def send(step):
        publisher.set(np.full(count, 0.001 * (step % 100)), zeros, zeros, gains, gains)
    return send, publisher.stop, None


def gate_sender(controller, profile, rate):
    """send(step) through a LowCmdHandler whose follow gate stays shut, its stop(), the handler"""
    from golem_upper_ik.config import load_controller_config
    from golem_upper_ik.core.low_cmd_handler import LowCmdHandler
    from golem_upper_ik.core.robot_model import RobotModel
    from golem_upper_ik.core.ros_io import StateSubscriber

    config = load_controller_config('h12', 'safety_split')
    config['topics']['low_cmd'] = TOPIC
    config['frequency']['pub_hz'] = rate
    pose = np.clip(np.full(len(profile.body_joints), 0.1), profile.q_low, profile.q_high)
    model = RobotModel(profile)
    model.state_subscriber = StateSubscriber(controller, profile)
    stream = StateStream(controller, profile, pose, 500.0)
    deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
    while model.state_subscriber.latest() is None:
        if time.monotonic() > deadline:
            raise RuntimeError('no /lowstate from the in-process stream')
        time.sleep(0.01)
    handler = LowCmdHandler(controller, profile, config, model)
    handler.start()
    count = len(profile.body_joints)
    zeros = np.zeros(count)

    def send(step):
        handler.set_joint_commands(q=pose + 0.3, dq=zeros, tau=zeros)

    def stop():
        handler.shutdown()
        stream.stop()
        model.shutdown()
    return send, stop, handler


def ik_loop(send, seconds, ik_hz, burn_ms):
    """Burn burn_ms of numpy work, then send a new command, at ik_hz for seconds."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((48, 48))
    b = rng.standard_normal((48, 48))
    ridge = 48.0 * np.eye(48)
    rhs = rng.standard_normal(48)
    period = 1.0 / ik_hz
    start = time.monotonic()
    next_at = start
    steps = 0
    while time.monotonic() - start < seconds:
        end = time.monotonic() + burn_ms / 1000.0
        while time.monotonic() < end:
            np.linalg.solve(a @ b + ridge, rhs)
        send(steps)
        steps += 1
        next_at += period
        time.sleep(max(0.0, next_at - time.monotonic()))
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rate', type=float, default=500.0)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--ik-hz', type=float, default=30.0)
    parser.add_argument('--burn-ms', type=float, default=15.0)
    parser.add_argument('--gate', action='store_true',
                        help='send through a LowCmdHandler whose follow gate stays shut')
    args = parser.parse_args()

    profile = load_profile('h12')
    capacity = int(args.rate * (args.seconds + 5.0)) + SPARE_SAMPLES
    rclpy.init()
    controller = rclpy.create_node('golem_rate_probe_controller')
    listener = rclpy.create_node('golem_rate_probe_listener')
    recorder = Recorder(listener, capacity)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(listener)
    executor.add_node(controller)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    sender = gate_sender if args.gate else publisher_sender
    send, stop, handler = sender(controller, profile, args.rate)
    deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
    while controller.count_subscribers(TOPIC) == 0:
        if time.monotonic() > deadline:
            print('no subscriber matched the command publisher', file=sys.stderr)
            return 1
        time.sleep(0.01)

    steps = ik_loop(send, args.seconds, args.ik_hz, args.burn_ms)
    gate_shut = None if handler is None else not handler.following
    stop()
    time.sleep(0.5)
    executor.shutdown()
    spin.join(timeout=2.0)

    received = recorder.times[:min(recorder.count, capacity)]
    if len(received) < 2:
        print(f'only {len(received)} samples received', file=sys.stderr)
        return 1
    gaps = np.diff(received) * 1000.0
    print(json.dumps({
        'target_hz': args.rate,
        'mean_hz': round((len(received) - 1) / (received[-1] - received[0]), 3),
        'p99_period_ms': round(float(np.percentile(gaps, 99)), 3),
        'max_gap_ms': round(float(gaps.max()), 3),
        'gate_shut': gate_shut,
    }))
    print(json.dumps({
        'received': recorder.count,
        'ik_steps': steps,
        'blas_threads': os.environ['OPENBLAS_NUM_THREADS'],
    }), file=sys.stderr)
    controller.destroy_node()
    listener.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
