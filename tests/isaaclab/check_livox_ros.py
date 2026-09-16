"""ROS-container half of CustomMsg interoperability test; exits nonzero on timeout."""
import time
import rclpy
from livox_ros_driver2.msg import CustomMsg

rclpy.init()
node = rclpy.create_node('isaac_livox_interop_check')
received = []
node.create_subscription(CustomMsg, '/livox/lidar', received.append, 5)
deadline = time.monotonic() + 30
while not received and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=.25)
try:
    assert received, 'no /livox/lidar CustomMsg received'
    msg = received[-1]
    assert msg.header.frame_id == 'lidar_link'
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (7, 123)
    assert msg.timebase == 7_000_000_123
    assert msg.point_num == 2
    assert [(p.x, p.y, p.z, p.offset_time, p.line) for p in msg.points] == [(1., 2., 3., 0, 0), (4., 5., 6., 500, 5)]
    print('PASS: native ROS decoded CycloneDDS Livox CustomMsg')
finally:
    node.destroy_node()
    rclpy.shutdown()
