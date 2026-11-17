"""Print a bag's relay timeline: first relayed frame, estop (first all-zero frame after one with gains), frames after it, and the measured joint nearest its estop limit then.

Usage: python3 timeline.py <bag> <config>   (times in s from the first recorded input)
"""
import json
import subprocess
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, '/tmp/bag_replay')
from compare import body  # noqa: E402

lim = json.loads(subprocess.run(
    ['/tmp/gs/install/golem_safety/lib/golem_safety/golem_safety_dump_config', 'h12', sys.argv[2]],
    capture_output=True, text=True, check=True).stdout)['limits']
lo, hi = np.array(lim['q_estop_low']), np.array(lim['q_estop_high'])
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(topics=['/lowcmd', '/lowstate', '/safety/lowcmd_lower_in', '/estop/triggered']))
t0, first, latch, after, state, trig = None, None, None, 0, None, None
margins = []
while r.has_next():
    topic, data, t = r.read_next()
    m = deserialize_message(data, get_message(types[topic]))
    if topic == '/safety/lowcmd_lower_in' and t0 is None:
        t0 = t
    if topic == '/estop/triggered' and m.data and trig is None:
        trig = t
    if topic == '/lowstate':
        q = np.array([m.motor_state[i].q for i in range(27)])
        state = (t, q)
        margins.append((t, float(np.min(np.minimum(q - lo, hi - q))), int(np.argmin(np.minimum(q - lo, hi - q))), q))
    if topic == '/lowcmd':
        gains = body(m)[:27, 4:].any()
        if gains and first is None:
            first = t
        if first is not None and not gains and latch is None:
            latch = t
        if latch is not None:
            after += 1
rel = lambda t: (t - t0) / 1e9 if t is not None else None
print(f'{sys.argv[1]}: first relayed frame {rel(first):.3f} s; estop {rel(latch)} s; frames after it {after}; '
      f'estop/triggered true first at {rel(trig)}')
if latch is not None:
    near = [m for m in margins if abs(m[0] - latch) < 50e6]
    worst = min(near, key=lambda m: m[1])
    print(f'  within 50 ms of the estop: smallest q margin to estop limit {worst[1]:.4f} rad at slot {worst[2]} '
          f'(q {worst[3][worst[2]]:.4f}, limits [{lo[worst[2]]:.4f}, {hi[worst[2]]:.4f}]) at {rel(worst[0]):.3f} s')
print(f'  smallest q margin over the whole bag: {min(m[1] for m in margins):.4f} rad '
      f'(slot {min(margins, key=lambda m: m[1])[2]} at {rel(min(margins, key=lambda m: m[1])[0]):.3f} s)')
