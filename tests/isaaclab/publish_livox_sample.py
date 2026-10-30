"""Isaac-container half of CustomMsg interoperability test (no GPU)."""
import os
import time
import numpy as np
from src.python.sensors.livox_dds import LivoxPublisher
from src.python.sensors.ros_sensor_bridge import POINT_DTYPE

domain = int(os.environ['ROS_DOMAIN_ID'])
assert 1 <= domain <= 232
publisher = LivoxPublisher(domain)
points = np.zeros(2, dtype=POINT_DTYPE)
points['x'], points['y'], points['z'] = [1., 4.], [2., 5.], [3., 6.]
points['timestamp'], points['line'] = [0, 500], [0, 5]
try:
    for _ in range(80):
        publisher.publish(7, 123, points)
        time.sleep(.1)
finally:
    publisher.close()
