"""Native ROS regression: a paused sim must not pause safety liveness.

Runs the installed golem_safety safety_node as the sim bringups do (use_sim_time
true, sim true) with no /clock publisher, and counts its /safety/heartbeat beats.
Its /lowcmd is remapped to a topic nothing reads, so the test commands nothing.
"""
import importlib.util
import os
import signal
import subprocess
import tempfile
import time
import unittest

# s to wait for the first beat (process start and discovery), then to count beats
FIRST_BEAT_TIMEOUT = 15.0
WINDOW = 1.5
# 10 Hz for WINDOW s gives 15; a paused clock gives none
MIN_BEATS = 5


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'requires ROS container')
class HeartbeatClockTests(unittest.TestCase):
    def test_heartbeat_advances_without_sim_clock(self):
        import rclpy
        from ament_index_python.packages import get_package_prefix
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Bool
        self.assertIn(int(os.environ.get('ROS_DOMAIN_ID', '0')), range(1, 233))
        safety_node = os.path.join(get_package_prefix('golem_safety'), 'lib', 'golem_safety', 'safety_node')
        rclpy.init()
        observer = rclpy.create_node('heartbeat_clock_observer')
        beats = []
        observer.create_subscription(
            Bool, '/safety/heartbeat', lambda m: beats.append((time.monotonic(), m.data)), 10)
        executor = SingleThreadedExecutor()
        executor.add_node(observer)
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen(
                [safety_node, '--ros-args', '-p', 'robot:=h12', '-p', 'config:=sim_safety_split',
                 '-p', 'sim:=true', '-p', 'use_sim_time:=true',
                 '-r', '/lowcmd:=/heartbeat_clock_test/lowcmd'],
                stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + FIRST_BEAT_TIMEOUT
                while not beats and time.monotonic() < deadline and process.poll() is None:
                    executor.spin_once(timeout_sec=.05)
                until = beats[0][0] + WINDOW if beats else 0.
                while time.monotonic() < until and process.poll() is None:
                    executor.spin_once(timeout_sec=.05)
                window = [data for stamp, data in beats if stamp <= until]
                clock_publishers = observer.count_publishers('/clock')
                exit_code = process.poll()
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                executor.shutdown()
                observer.destroy_node()
                rclpy.shutdown()
            log.seek(0)
            output = f' (safety_node exit code {exit_code}):\n{log.read()}'
        self.assertTrue(beats, f'No /safety/heartbeat within {FIRST_BEAT_TIMEOUT} s{output}')
        self.assertEqual(clock_publishers, 0,
                         'A /clock publisher is on this domain: the test needs a paused clock')
        self.assertGreaterEqual(len(window), MIN_BEATS, f'Paused /clock suppressed safety heartbeat{output}')
        self.assertTrue(all(window), f'Heartbeat false{output}')


if __name__ == '__main__':
    unittest.main()
