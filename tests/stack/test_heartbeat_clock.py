"""Native ROS regression: a paused sim must not pause safety liveness.

Runs the installed golem_safety safety_node as the sim bringups do (use_sim_time
true, sim true) with no /clock publisher, and counts its /safety/heartbeat beats.
The node beats only once relaying can begin, so the test feeds it an H1-2
/lowstate (mode_machine 6) at 100 Hz. Its /lowstate and /lowcmd are remapped to
topics of the test's own, so nothing else reads or writes them.
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
STATE_PERIOD = 0.01  # s, wall clock: the fed /lowstate at 100 Hz
H12_MODE_MACHINE = 6


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'requires ROS container')
class HeartbeatClockTests(unittest.TestCase):
    def test_heartbeat_advances_without_sim_clock(self):
        import rclpy
        from ament_index_python.packages import get_package_prefix
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Bool
        from unitree_hg.msg import LowState
        self.assertIn(int(os.environ.get('ROS_DOMAIN_ID', '0')), range(1, 233))
        safety_node = os.path.join(get_package_prefix('golem_safety'), 'lib', 'golem_safety', 'safety_node')
        rclpy.init()
        observer = rclpy.create_node('heartbeat_clock_observer')
        beats = []
        observer.create_subscription(
            Bool, '/safety/heartbeat', lambda m: beats.append((time.monotonic(), m.data)), 10)
        state = LowState(mode_machine=H12_MODE_MACHINE)
        state_pub = observer.create_publisher(LowState, '/heartbeat_clock_test/lowstate', 10)
        # a wall timer: the observer does not use sim time either
        observer.create_timer(STATE_PERIOD, lambda: state_pub.publish(state))
        executor = SingleThreadedExecutor()
        executor.add_node(observer)
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen(
                [safety_node, '--ros-args', '-p', 'robot:=h12', '-p', 'config:=sim_safety_split',
                 '-p', 'sim:=true', '-p', 'use_sim_time:=true',
                 '-r', '/lowcmd:=/heartbeat_clock_test/lowcmd',
                 '-r', '/lowstate:=/heartbeat_clock_test/lowstate'],
                stdout=log, stderr=subprocess.STDOUT)
            exit_code = None
            try:
                deadline = time.monotonic() + FIRST_BEAT_TIMEOUT
                while not beats and time.monotonic() < deadline and process.poll() is None:
                    executor.spin_once(timeout_sec=.05)
                until = beats[0][0] + WINDOW if beats else 0.
                while time.monotonic() < until and process.poll() is None:
                    executor.spin_once(timeout_sec=.05)
                window = [data for stamp, data in beats if stamp <= until]
                clock_publishers = observer.count_publishers('/clock')
                heartbeat_publishers = observer.count_publishers('/safety/heartbeat')
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
        self.assertIsNone(exit_code, f'safety_node exited before the end of the test{output}')
        self.assertEqual(clock_publishers, 0,
                         'A /clock publisher is on this domain: the test needs a paused clock')
        self.assertEqual(heartbeat_publishers, 1,
                         'The beats must come from this safety_node alone: another /safety/heartbeat '
                         'publisher is on this domain')
        self.assertGreaterEqual(len(window), MIN_BEATS, f'Paused /clock suppressed safety heartbeat{output}')
        self.assertTrue(all(window), f'Heartbeat false{output}')


if __name__ == '__main__':
    unittest.main()
