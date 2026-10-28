#!/usr/bin/env python3
"""Measure safety_node's relay on a running stack: output rate, input-to-output latency.

Subscribes, and never publishes, to the upper-body controller's input
(/safety/lowcmd_upper_in) and to safety_node's output (/lowcmd, or /arm_sdk for the G1's
arm_sdk mode), and records each sample's arrival time on the monotonic clock with the q of the
robot's upper slots (from golem_safety's robots/<robot>.yaml). Recording starts --settle s
after both topics have delivered a sample and lasts --seconds: the publishers on a domain can
stall for tens of ms while a participant that joins it, this probe's included, is discovered.

- Output rate: the mean, the 99th percentile of the arrival interval and the longest gap.
- Latency: each output sample is matched to the newest input received before it whose
  upper-slot q equals the output's exactly, and the latency is the time between their
  arrivals here: safety_node's wait for its next cycle plus its own DDS hop. While the
  controller repeats one q, that newest input can be a later copy than the one safety_node
  took, so a latency can read short.
- Unmatched outputs are counted, not timed. `no_equal_input`: no input received so far
  carries the output's q (the zero-gain stream before the input engages, or a sample with a q
  that safety_node clipped to its limits; it relays q unchanged otherwise). `repeated`: the
  newest input with that q was matched already, since an input is matched at most once
  (safety_node sending again a command that the controller has not renewed).

The JSON on stdout gives the output's and the input's rate, how many distinct q the input
carried (the controller's steps), the latency and the unmatched counts, in the units named by
the keys. ROS domain 0, the real robots' command bus, is refused, and so is an unset
ROS_DOMAIN_ID, with which ROS joins domain 0.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

DISCOVERY_TIMEOUT_S = 10.0
RECORD_DEPTH = 10000


@dataclass(frozen=True)
class Matching:
    latencies: list  # s, one per matched output
    no_equal_input: int
    repeated: int

    @property
    def unmatched(self) -> int:
        return self.no_equal_input + self.repeated


def match_outputs(inputs, outputs) -> Matching:
    """Match each output sample to the newest input before it with the same q.

    inputs and outputs are (arrival time in s, upper-slot q tuple), each in arrival order.
    """
    newest = {}  # q -> index of the newest input with that q received so far
    matched = set()
    latencies = []
    no_equal_input = 0
    repeated = 0
    next_input = 0
    for arrival, q in outputs:
        while next_input < len(inputs) and inputs[next_input][0] <= arrival:
            newest[inputs[next_input][1]] = next_input
            next_input += 1
        index = newest.get(q)
        if index is None:
            no_equal_input += 1
        elif index in matched:
            repeated += 1
        else:
            matched.add(index)
            latencies.append(arrival - inputs[index][0])
    return Matching(latencies, no_equal_input, repeated)


def percentile(values, pct: float) -> float:
    """Linear interpolation between the closest ranks, as numpy.percentile's default"""
    ordered = sorted(values)
    if not ordered:
        raise ValueError('no values')
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def rate_stats(times) -> dict:
    """Rate of arrivals at times (s, ascending): mean, p99 interval and longest gap"""
    if len(times) < 2:
        raise ValueError(f'{len(times)} samples: a rate needs at least 2')
    intervals_ms = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
    return {
        'samples': len(times),
        'mean_hz': (len(times) - 1) / (times[-1] - times[0]),
        'p99_interval_ms': percentile(intervals_ms, 99.0),
        'max_gap_ms': max(intervals_ms),
    }


def latency_stats(latencies) -> dict:
    """Median, p99 and longest of the matched latencies (s), in ms"""
    if not latencies:
        raise ValueError('no output sample matched an input')
    latencies_ms = [latency * 1000.0 for latency in latencies]
    return {
        'matched': len(latencies_ms),
        'median_ms': percentile(latencies_ms, 50.0),
        'p99_ms': percentile(latencies_ms, 99.0),
        'max_ms': max(latencies_ms),
    }


def domain_refusal(value):
    """Why this process must not join the ROS domain ROS_DOMAIN_ID=value, or None"""
    if value is None or not value.strip():
        return ('ROS_DOMAIN_ID is unset, so ROS would join domain 0, the real robots\' command '
                'bus; export the stack\'s domain')
    try:
        domain = int(value)
    except ValueError:
        return f'ROS_DOMAIN_ID {value!r} is not a domain number'
    if domain == 0:
        return 'ROS_DOMAIN_ID is 0, the real robots\' command bus; export the stack\'s domain'
    return None


def upper_slots(robot: str) -> range:
    """The robot's upper-body slots, from golem_safety's installed robots/<robot>.yaml"""
    import yaml
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory('golem_safety'), 'robots', f'{robot}.yaml')
    with open(path, encoding='utf-8') as handle:
        first, last = yaml.safe_load(handle)['upper_slots']
    return range(first, last + 1)


def record(slots: range, input_topic: str, output_topic: str, settle: float, seconds: float):
    """(inputs, outputs, window start, window end): every sample's arrival and upper-slot q"""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from unitree_hg.msg import LowCmd

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     history=HistoryPolicy.KEEP_LAST, depth=RECORD_DEPTH)
    inputs, outputs = [], []

    def recorder(samples):
        def on_message(msg):
            arrival = time.monotonic()
            motors = msg.motor_cmd
            samples.append((arrival, tuple(motors[slot].q for slot in slots)))
        return on_message

    rclpy.init()
    node = rclpy.create_node('relay_latency_probe')
    node.create_subscription(LowCmd, input_topic, recorder(inputs), qos)
    node.create_subscription(LowCmd, output_topic, recorder(outputs), qos)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
        while not (inputs and outputs):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f'no sample within {DISCOVERY_TIMEOUT_S:g} s on '
                    + ' or '.join(topic for topic, samples in ((input_topic, inputs),
                                                               (output_topic, outputs))
                                  if not samples))
            executor.spin_once(timeout_sec=0.05)
        start = time.monotonic() + settle
        end = start + seconds
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return inputs, outputs, start, end


def rounded(stats: dict) -> dict:
    return {key: round(value, 3) if isinstance(value, float) else value
            for key, value in stats.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--robot', required=True, choices=('h12', 'g1'))
    parser.add_argument('--output', default='/lowcmd', choices=('/lowcmd', '/arm_sdk'),
                        help="safety_node's output; /arm_sdk in the G1's arm_sdk mode")
    parser.add_argument('--input', default='/safety/lowcmd_upper_in')
    parser.add_argument('--settle', type=float, default=2.0,
                        help='s from the first samples to the start of the recording')
    parser.add_argument('--seconds', type=float, default=10.0)
    args = parser.parse_args()

    refusal = domain_refusal(os.environ.get('ROS_DOMAIN_ID'))
    if refusal:
        print(f'relay_latency_probe refuses to start: {refusal}', file=sys.stderr)
        return 2

    try:
        inputs, outputs, start, end = record(upper_slots(args.robot), args.input, args.output,
                                             args.settle, args.seconds)
        window_outputs = [sample for sample in outputs if start <= sample[0] < end]
        window_inputs = [sample for sample in inputs if start <= sample[0] < end]
        matching = match_outputs([sample for sample in inputs if sample[0] < end],
                                 window_outputs)
        result = {
            'robot': args.robot,
            'domain': int(os.environ['ROS_DOMAIN_ID']),
            'seconds': args.seconds,
            'output': {'topic': args.output,
                       **rounded(rate_stats([t for t, _ in window_outputs]))},
            'input': {'topic': args.input,
                      **rounded(rate_stats([t for t, _ in window_inputs]))},
            'input_distinct_q': len({q for _, q in window_inputs}),
            'latency': rounded(latency_stats(matching.latencies)),
            'unmatched': {'total': matching.unmatched,
                          'no_equal_input': matching.no_equal_input,
                          'repeated': matching.repeated},
        }
    except (RuntimeError, ValueError) as error:
        print(f'relay_latency_probe: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
