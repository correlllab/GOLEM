"""Print the header values and counts of every LowCmd/LowState topic in a bag."""
import sys
from collections import Counter
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id='sqlite3'),
       rosbag2_py.ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = [t for t, ty in types.items() if ty.startswith('unitree_hg/msg/Low')]
r.set_filter(rosbag2_py.StorageFilter(topics=want))
hdr, spare = {t: Counter() for t in want}, {t: Counter() for t in want}
while r.has_next():
    topic, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types[topic]))
    hdr[topic][(m.mode_machine, m.mode_pr)] += 1
    arr = m.motor_cmd if hasattr(m, 'motor_cmd') else m.motor_state
    spare[topic][any(abs(arr[i].q) > 0 or getattr(arr[i], 'kp', 0) for i in range(27, 35))] += 1
for t in want:
    print(t, '(mode_machine, mode_pr):', dict(hdr[t]), ' nonzero slots 27-34:', dict(spare[t]))
