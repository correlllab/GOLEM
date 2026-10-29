'''The velocity smoother with config/g1_nav2.yaml and the launch file's remappings

It is the only publisher of /cmd_vel, so it is what enforces the caps.
Out-of-range commands on cmd_vel_nav (where the controller and the behaviors
publish) have to come out on /cmd_vel clamped to the caps, with no sideways
motion. When they stop, the smoother may hold the last command for less than
the walking node's own stale-command timeout, then has to ramp to zero and go
quiet. About 15 s.
'''

import importlib.util
import os
import threading
import time

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

import stack

SHARE = get_package_share_directory('g1_navigation')
CAPS = {'vx': (-0.3, 0.5), 'vy': (0.0, 0.0), 'wz': (-0.6, 0.6)}
WALKING_NODE_TIMEOUT = 0.5  # s, g1_homie stops on a command this old


def _smoother_remappings():
    spec = importlib.util.spec_from_file_location(
        'g1_navigation_launch', os.path.join(SHARE, 'launch', 'g1_navigation.launch.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    [remappings] = [r for _, executable, r in module.NAV2_NODES
                    if executable == 'velocity_smoother']
    return remappings


class Probe(Node):

    def __init__(self):
        super().__init__('velocity_smoother_probe')
        self.lock = threading.Lock()
        self.out = []  # (receive time, Twist)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 100)
        self.pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.change_state = self.create_client(ChangeState, '/velocity_smoother/change_state')

    def _on_cmd(self, msg):
        with self.lock:
            self.out.append((time.time(), msg))

    def take(self):
        with self.lock:
            out, self.out = self.out, []
        return out


@pytest.fixture
def smoother(tmp_path):
    stack.use_loopback_dds(str(tmp_path))
    procs = stack.Processes(str(tmp_path))
    rclpy.init()
    probe = Probe()
    executor = SingleThreadedExecutor()
    executor.add_node(probe)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        executable = os.path.join(get_package_prefix('nav2_velocity_smoother'), 'lib',
                                  'nav2_velocity_smoother', 'velocity_smoother')
        remaps = [arg for a, b in _smoother_remappings() for arg in ('-r', f'{a}:={b}')]
        procs.start('velocity_smoother', [
            executable, '--ros-args', '--params-file',
            os.path.join(SHARE, 'config', 'g1_nav2.yaml'), *remaps])
        assert probe.change_state.wait_for_service(timeout_sec=30.0), procs.tail(
            'velocity_smoother')
        for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
            future = probe.change_state.call_async(ChangeState.Request(
                transition=Transition(id=transition)))
            deadline = time.time() + 10.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.05)
            assert future.done() and future.result().success, procs.tail('velocity_smoother')
        yield procs, probe
    finally:
        executor.shutdown()
        probe.destroy_node()
        rclpy.shutdown()
        procs.stop_all()


def _publish(probe, twist, seconds):
    end = time.time() + seconds
    while time.time() < end:
        probe.pub.publish(twist)
        time.sleep(0.05)  # 20 Hz, like the controller
    return time.time()


def test_clamped_to_the_caps_and_stops_when_the_input_stops(smoother):
    procs, probe = smoother
    tol = 1e-6
    last = None
    for vx, vy, wz in ((1.5, 0.8, 2.0), (-1.5, -0.8, -2.0)):
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
        cmd.linear.z = cmd.angular.x = cmd.angular.y = 1.0
        stopped = _publish(probe, cmd, 3.0)
        out = probe.take()
        assert len(out) > 20, procs.tail('velocity_smoother')
        for _, msg in out:
            assert CAPS['vx'][0] - tol <= msg.linear.x <= CAPS['vx'][1] + tol, msg
            assert abs(msg.linear.y) < tol, msg
            assert CAPS['wz'][0] - tol <= msg.angular.z <= CAPS['wz'][1] + tol, msg
            assert abs(msg.linear.z) < tol and abs(msg.angular.x) < tol and abs(
                msg.angular.y) < tol, msg
        # ramped all the way to the cap
        last = out[-1][1]
        cap = 1 if vx > 0 else 0
        assert abs(last.linear.x - CAPS['vx'][cap]) < tol, last
        assert abs(last.angular.z - CAPS['wz'][cap]) < tol, last

    # the input stops: the last command is held briefly, then ramps to zero and stops
    time.sleep(3.0)
    out = probe.take()
    assert out, 'the smoother published nothing after the input stopped'
    held = [t - stopped for t, msg in out if abs(msg.linear.x - last.linear.x) < tol
            and abs(msg.angular.z - last.angular.z) < tol]
    print(f'last command held {max(held, default=0.0):.2f} s; zero after '
          f'{out[-1][0] - stopped:.2f} s, {len(out)} messages')
    assert max(held, default=0.0) < WALKING_NODE_TIMEOUT, held
    final = out[-1][1]
    assert abs(final.linear.x) < tol and abs(final.angular.z) < tol, final
    assert out[-1][0] - stopped < 2.0  # quiet after that
