"""Native ROS regression: a paused sim must not pause safety liveness."""
import importlib.util
import os
import time
import unittest
from unittest.mock import patch, MagicMock


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'requires ROS container')
class HeartbeatClockTests(unittest.TestCase):
    def test_heartbeat_advances_without_sim_clock(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Bool
        from h12_safety_layer.ros2.safety_node import SafetyNode
        self.assertIn(int(os.environ.get('ROS_DOMAIN_ID', '0')), range(1, 233))
        rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
        layer = MagicMock()
        layer.estop_triggered = False
        with patch('h12_safety_layer.ros2.safety_node.SafetyLayer', return_value=layer):
            node = SafetyNode('sim_safety_split.yaml')
        observer = rclpy.create_node('heartbeat_clock_observer')
        received = []
        observer.create_subscription(Bool, '/safety/heartbeat', lambda m: received.append(m.data), 10)
        executor = SingleThreadedExecutor()
        executor.add_node(node); executor.add_node(observer)
        try:
            until = time.monotonic() + 1.5
            while time.monotonic() < until:
                executor.spin_once(timeout_sec=.05)
            self.assertGreaterEqual(len(received), 5, 'Paused /clock suppressed safety heartbeat')
            self.assertTrue(all(received))
        finally:
            executor.shutdown(); observer.destroy_node(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
