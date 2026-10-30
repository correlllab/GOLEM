#!/usr/bin/env python3
"""Measure whether rclpy holds a LowCmd publish rate beside IK-sized CPU load.

One process, three threads. A publisher thread sends a preallocated
unitree_hg/msg/LowCmd on /golem_probe/lowcmd against a monotonic deadline. The
main thread repeats a 30 Hz "IK step" that burns CPU for --burn-ms of every
period. A second node, spun by its own executor thread, subscribes with a QoS
that keeps every sample and records each receive time.

The JSON on stdout is the receive side: the achieved rate, the 99th percentile
of the receive period and the longest gap, in the units named by the keys.
Publisher-side numbers and sample counts go to stderr.

--burn numpy (default) loads the main thread with small dense linear algebra.
--burn gil loads it with pure Python arithmetic, which keeps the GIL until the
interpreter's switch interval (--switch-interval-ms) forces a hand-over, the
worst case for the publisher thread.
"""
import argparse
import json
import os
import sys
import threading
import time

# OpenBLAS starts one busy-waiting worker per core by default, and their spin
# adds several ms to tens of ms of jitter to the publisher thread. Set before
# numpy loads BLAS.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from unitree_hg.msg import LowCmd  # noqa: E402

TOPIC = '/golem_probe/lowcmd'
UPPER_SLOTS = range(12, 29)
DISCOVERY_TIMEOUT_S = 10.0
SPARE_SAMPLES = 1000


class Recorder(Node):
    """Subscribes with reliable, deep keep-last QoS and stamps every receive."""

    def __init__(self, capacity):
        super().__init__('golem_probe_sub')
        self.times = np.zeros(capacity)
        self.count = 0
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10000)
        self.create_subscription(LowCmd, TOPIC, self._on_msg, qos)

    def _on_msg(self, _msg):
        if self.count < len(self.times):
            self.times[self.count] = time.monotonic()
        self.count += 1


class Publisher(Node):
    """Owns the publisher and one preallocated message."""

    def __init__(self):
        super().__init__('golem_probe_pub')
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(LowCmd, TOPIC, qos)
        self.msg = LowCmd()
        self.times = np.zeros(0)
        self.count = 0
        self.max_publish_s = 0.0
        self.max_late_s = 0.0

    def run(self, rate_hz, stop, capacity):
        """Publish at rate_hz until stop is set, against absolute deadlines."""
        self.times = np.zeros(capacity)
        period = 1.0 / rate_hz
        motor_cmd = self.msg.motor_cmd
        pub = self.pub
        msg = self.msg
        next_at = time.monotonic()
        while not stop.is_set():
            sent_at = time.monotonic()
            self.max_late_s = max(self.max_late_s, sent_at - next_at)
            if self.count < capacity:
                self.times[self.count] = sent_at
            phase = 0.01 * (self.count % 628)
            for slot in UPPER_SLOTS:
                motor_cmd[slot].q = phase
            msg.crc = self.count & 0xFFFFFFFF
            pub.publish(msg)
            self.max_publish_s = max(self.max_publish_s,
                                     time.monotonic() - sent_at)
            self.count += 1
            next_at += period
            time.sleep(max(0.0, next_at - time.monotonic()))


def make_load(burn_kind):
    """Return a callable that keeps one CPU thread busy until a deadline."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((48, 48))
    b = rng.standard_normal((48, 48))
    ridge = 48.0 * np.eye(48)
    rhs = rng.standard_normal(48)

    def numpy_load(end):
        while time.monotonic() < end:
            np.linalg.solve(a @ b + ridge, rhs)

    def gil_load(end):
        # Pure Python arithmetic never releases the GIL voluntarily.
        while time.monotonic() < end:
            total = 0.0
            for i in range(2000):
                total += i * 0.5
    return numpy_load if burn_kind == 'numpy' else gil_load


def ik_loop(seconds, ik_hz, burn_ms, load):
    """Run IK-sized bursts at ik_hz on the calling thread for seconds."""
    period = 1.0 / ik_hz
    start = time.monotonic()
    next_at = start
    steps = 0
    while time.monotonic() - start < seconds:
        load(time.monotonic() + burn_ms / 1000.0)
        steps += 1
        next_at += period
        time.sleep(max(0.0, next_at - time.monotonic()))
    return steps


def periods_ms(times):
    return np.diff(times) * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rate', type=float, default=500.0)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--ik-hz', type=float, default=30.0)
    parser.add_argument('--burn-ms', type=float, default=15.0)
    parser.add_argument('--burn', choices=('numpy', 'gil'), default='numpy')
    parser.add_argument('--switch-interval-ms', type=float, default=None,
                        help='sys.setswitchinterval; default is the interpreter'
                        "'s (5 ms)")
    args = parser.parse_args()

    if args.switch_interval_ms is not None:
        sys.setswitchinterval(args.switch_interval_ms / 1000.0)

    capacity = int(args.rate * (args.seconds + 5.0)) + SPARE_SAMPLES
    rclpy.init()
    publisher = Publisher()
    recorder = Recorder(capacity)
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
    while publisher.pub.get_subscription_count() == 0:
        if time.monotonic() > deadline:
            print('no subscriber matched the probe publisher', file=sys.stderr)
            return 1
        time.sleep(0.01)

    stop = threading.Event()
    pub_thread = threading.Thread(
        target=publisher.run, args=(args.rate, stop, capacity))
    pub_thread.start()
    steps = ik_loop(args.seconds, args.ik_hz, args.burn_ms,
                    make_load(args.burn))
    stop.set()
    pub_thread.join()
    time.sleep(0.5)
    executor.shutdown()
    spin.join(timeout=2.0)

    received = recorder.times[:min(recorder.count, capacity)]
    if len(received) < 2:
        print(f'only {len(received)} samples received', file=sys.stderr)
        return 1
    gaps = periods_ms(received)
    stats = {
        'target_hz': args.rate,
        'mean_hz': round((len(received) - 1) / (received[-1] - received[0]), 3),
        'p99_period_ms': round(float(np.percentile(gaps, 99)), 3),
        'max_gap_ms': round(float(gaps.max()), 3),
    }
    print(json.dumps(stats))

    sent = publisher.times[:min(publisher.count, capacity)]
    sent_gaps = periods_ms(sent)
    worst = np.argsort(gaps)[-5:][::-1]
    print(json.dumps({
        'worst_gaps_ms_at_s': [
            [round(float(gaps[i]), 2), round(float(received[i] - received[0]), 3)]
            for i in worst],
        'max_wake_late_ms': round(publisher.max_late_s * 1000.0, 3),
        'max_publish_call_ms': round(publisher.max_publish_s * 1000.0, 3),
        'published': publisher.count,
        'received': recorder.count,
        'ik_steps': steps,
        'pub_p99_period_ms': round(float(np.percentile(sent_gaps, 99)), 3),
        'pub_max_gap_ms': round(float(sent_gaps.max()), 3),
        'reliability': 'reliable',
        'blas_threads': os.environ['OPENBLAS_NUM_THREADS'],
        'burn': args.burn,
        'switch_interval_s': sys.getswitchinterval(),
    }), file=sys.stderr)
    publisher.destroy_node()
    recorder.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
