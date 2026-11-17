"""Check that a bag's /lowcmd is its safety inputs passed through unchanged.

Usage: python3 compare.py <bag> [<bag> ...]

For every /lowcmd in the bag, builds the expected command from any
/safety/lowcmd_lower_in (slots 0-11) and /safety/lowcmd_upper_in (slots 12-26)
received from BEFORE_NS before it to AFTER_NS after it (the relay can lag, and
the recorder can log an input after the /lowcmd it produced), and compares the 27 body slots field by field (mode, q,
dq, tau, kp, kd), the header and slots 27-34. Prints the counts and, for frames
that match no input in the window, which joints and fields differ and by how much.
"""
import sys
from collections import Counter

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

FIELDS = ('mode', 'q', 'dq', 'tau', 'kp', 'kd')
BEFORE_NS, AFTER_NS = 200_000_000, 60_000_000  # input window around a /lowcmd, by receive time


def body(msg):
    """Return the 35 slots' (mode, q, dq, tau, kp, kd) as a float32 array."""
    return np.array([[m.mode, m.q, m.dq, m.tau, m.kp, m.kd] for m in msg.motor_cmd],
                    dtype=np.float32)


def read(uri):
    """Yield (topic, receive time ns, message) for the LowCmd topics of a bag."""
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=uri, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    topics = ['/lowcmd', '/safety/lowcmd_lower_in', '/safety/lowcmd_upper_in']
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    while r.has_next():
        topic, data, t = r.read_next()
        yield topic, t, deserialize_message(data, get_message(types[topic]))


def compare(uri):
    """Print how the bag's /lowcmd relates to its inputs."""
    lower, upper, outs = [], [], []
    for topic, t, msg in read(uri):
        if topic == '/safety/lowcmd_lower_in':
            lower.append((t, body(msg)))
        elif topic == '/safety/lowcmd_upper_in':
            upper.append((t, body(msg)))
        else:
            outs.append((t, msg.mode_machine, msg.mode_pr, body(msg)))
    lt = np.array([x[0] for x in lower])
    ut = np.array([x[0] for x in upper])
    counts, diffs, worst = Counter(), Counter(), {}
    for t, mm, pr, out in outs:
        lw = [lower[i][1] for i in np.nonzero((lt >= t - BEFORE_NS) & (lt <= t + AFTER_NS))[0]]
        uw = [upper[i][1] for i in np.nonzero((ut >= t - BEFORE_NS) & (ut <= t + AFTER_NS))[0]]
        if not out[:27, 4:].any():
            counts['all gains zero (zero or estop command)'] += 1
            continue
        if not lw or not uw:
            counts['no input of each kind within the window'] += 1
            continue
        counts['header (%d,%d)' % (mm, pr)] += 1
        counts['slots 27-34 zero' if not out[27:].any() else 'slots 27-34 nonzero'] += 1
        cands = [np.concatenate([lo[:12], up[12:27]]) for lo in lw for up in uw]
        if any(np.array_equal(out[:27], c) for c in cands):
            counts['body identical to an input within the window'] += 1
            continue
        counts['body differs from every input within the window'] += 1
        best = min(cands, key=lambda c: np.abs(out[:27] - c).sum())
        for j, f in zip(*np.nonzero(out[:27] != best)):
            key = (int(j), FIELDS[f])
            diffs[key] += 1
            worst[key] = max(worst.get(key, 0.0), float(abs(out[j, f] - best[j, f])))
    print(f'== {uri}: {len(outs)} /lowcmd, {len(lower)} lower, {len(upper)} upper')
    for k, v in sorted(counts.items()):
        print(f'  {k}: {v}')
    for (j, f), n in diffs.most_common(20):
        print(f'  differs: slot {j} {f}: {n} frames, largest change {worst[(j, f)]:.4g}')

if __name__ == '__main__':
    for uri in sys.argv[1:]:
        compare(uri)
