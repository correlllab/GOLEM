#!/usr/bin/env python3
"""Measure CommandPublisher's publish rate beside IK-sized CPU load.

One process. A CommandPublisher (golem_upper_ik.core.ros_io) sends the H1-2 profile's
commands on /safety/rate_probe_upper_in at --rate. The main thread runs 30 Hz "IK steps"
that burn --burn-ms of numpy work and then set() a new command, as the controller does. A
second node, spun by an executor thread, subscribes with a QoS that keeps every sample and
records each receive time.

The JSON on stdout is the receive side: the achieved rate, the 99th percentile of the
receive period and the longest gap, in the units named by the keys. Counts go to stderr.
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
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from unitree_hg.msg import LowCmd  # noqa: E402

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


def ik_loop(publisher, upper_count, seconds, ik_hz, burn_ms):
    """Burn burn_ms of numpy work, then set a new command, at ik_hz for seconds."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((48, 48))
    b = rng.standard_normal((48, 48))
    ridge = 48.0 * np.eye(48)
    rhs = rng.standard_normal(48)
    zeros = np.zeros(upper_count)
    gains = np.full(upper_count, 100.0)
    period = 1.0 / ik_hz
    start = time.monotonic()
    next_at = start
    steps = 0
    while time.monotonic() - start < seconds:
        end = time.monotonic() + burn_ms / 1000.0
        while time.monotonic() < end:
            np.linalg.solve(a @ b + ridge, rhs)
        publisher.set(np.full(upper_count, 0.001 * (steps % 100)), zeros, zeros, gains, gains)
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
    args = parser.parse_args()

    profile = load_profile('h12')
    upper_count = len(profile.upper_slots)
    capacity = int(args.rate * (args.seconds + 5.0)) + SPARE_SAMPLES
    rclpy.init()
    controller = rclpy.create_node('golem_rate_probe_controller')
    listener = rclpy.create_node('golem_rate_probe_listener')
    recorder = Recorder(listener, capacity)
    executor = SingleThreadedExecutor()
    executor.add_node(listener)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    publisher = CommandPublisher(controller, TOPIC, args.rate, profile)
    deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
    while controller.count_subscribers(TOPIC) == 0:
        if time.monotonic() > deadline:
            print('no subscriber matched the command publisher', file=sys.stderr)
            return 1
        time.sleep(0.01)

    steps = ik_loop(publisher, upper_count, args.seconds, args.ik_hz, args.burn_ms)
    publisher.stop()
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
