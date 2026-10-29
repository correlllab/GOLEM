'''Nav2 end to end with odom_source:=sim_gt against a fake G1: no robot, no sim.

fake_g1.py walks as /cmd_vel says, leans like HOMIE (pelvis and waist) and
ray-casts the upside-down MID-360 in a 7 x 4 m room with a 0.6 m box on the
straight line to the goal. robot_state_publisher, slam_toolbox and Nav2 run as
on the robot (the last two from g1_navigation.launch.py). A NavigateToPose goal
on the far side of the box has to succeed, the footprint must never overlap
the box, every /cmd_vel must respect the caps, and the SLAM map may hold no
obstacle that is not in the room (the floor, seen through a band that tilts
with the pelvis, would show up as one). Takes about a minute.
'''

import ast
import math
import os
import threading
import time

import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation

import fake_g1
import stack

GOAL = (4.0, 0.0)
CAPS = {'vx': (-0.3, 0.5), 'vy': (0.0, 0.0), 'wz': (-0.6, 0.6)}
STARTUP_TIMEOUT = 90.0
GOAL_TIMEOUT = 150.0
MAP_TOLERANCE = 0.15  # m, occupied map cells to the nearest box of the room


def _config_footprint():
    path = os.path.join(get_package_share_directory('g1_navigation'), 'config', 'g1_nav2.yaml')
    with open(path) as f:
        config = yaml.safe_load(f)
    return ast.literal_eval(
        config['local_costmap']['local_costmap']['ros__parameters']['footprint'])


class Monitor(Node):

    def __init__(self):
        super().__init__('nav2_smoke_monitor')
        self.lock = threading.Lock()
        self.cmds = []  # (vx, vy, vz, wx, wy, wz)
        self.poses = []  # (t, x, y, yaw) of the fake's pelvis, in odom
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 100)
        self.create_subscription(PoseStamped, '/fake_g1/pose', self._on_pose, 100)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map = None
        self.create_subscription(OccupancyGrid, '/map', self._on_map, latched)
        self.navigate = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def _on_map(self, msg):
        with self.lock:
            self.map = msg

    def _on_cmd(self, msg):
        with self.lock:
            self.cmds.append((msg.linear.x, msg.linear.y, msg.linear.z,
                              msg.angular.x, msg.angular.y, msg.angular.z))

    def _on_pose(self, msg):
        q = msg.pose.orientation
        yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        with self.lock:
            self.poses.append((time.time(), msg.pose.position.x, msg.pose.position.y, yaw))

    def pose(self):
        with self.lock:
            return self.poses[-1] if self.poses else None


def _wait(future, timeout):
    deadline = time.time() + timeout
    while not future.done() and time.time() < deadline:
        time.sleep(0.1)
    return future.done()


@pytest.fixture
def stack_up(tmp_path):
    stack.use_loopback_dds(str(tmp_path))
    procs = stack.Processes(str(tmp_path))
    rclpy.init()
    monitor = Monitor()
    executor = MultiThreadedExecutor()
    executor.add_node(monitor)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        procs.robot_state_publisher()
        procs.fake_g1('--odom', '--sway', '--waist', str(fake_g1.HOMIE_WAIST))
        procs.navigation('odom_source:=sim_gt')
        yield procs, monitor
    finally:
        executor.shutdown()
        monitor.destroy_node()
        rclpy.shutdown()
        procs.stop_all()


def _send_goal(procs, monitor):
    '''Send the goal once Nav2 accepts one (the navigator activates after SLAM's first map)'''
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x, goal.pose.pose.position.y = GOAL
    goal.pose.pose.orientation.w = 1.0
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        procs.check_alive()
        if monitor.navigate.wait_for_server(timeout_sec=1.0):
            goal.pose.header.stamp = monitor.get_clock().now().to_msg()
            sent = monitor.navigate.send_goal_async(goal)
            if _wait(sent, 10.0) and sent.result().accepted:
                return sent.result()
        time.sleep(2.0)
    pytest.fail(f'Nav2 never accepted the goal:\n{procs.tail("navigation")}')


def test_goal_across_the_room_around_the_box(stack_up):
    procs, monitor = stack_up
    handle = _send_goal(procs, monitor)
    start = monitor.pose()
    result = handle.get_result_async()
    finished = _wait(result, GOAL_TIMEOUT)
    procs.check_alive()
    end = monitor.pose()
    with monitor.lock:
        poses, cmds, grid = list(monitor.poses), list(monitor.cmds), monitor.map
    publishers = sorted(i.node_name for i in monitor.get_publishers_info_by_topic('/cmd_vel'))

    travelled = math.dist(start[1:3], end[1:3])
    to_goal = math.dist(end[1:3], GOAL)
    print(f'start {start[1:]}, end {end[1:]}, travelled {travelled:.2f} m, '
          f'{to_goal:.2f} m from the goal, {len(cmds)} /cmd_vel')
    assert finished, f'no result after {GOAL_TIMEOUT} s, {to_goal:.2f} m from the goal'
    assert result.result().status == GoalStatus.STATUS_SUCCEEDED, (
        f'status {result.result().status}, {to_goal:.2f} m from the goal:\n'
        f'{procs.tail("navigation", 80)}')
    assert to_goal < 0.4

    # the footprint never overlaps the box
    footprint = _config_footprint()
    hits = [p for p in poses if fake_g1.footprint_hits_box(p[1], p[2], p[3], footprint,
                                                           fake_g1.OBSTACLE)]
    assert not hits, f'{len(hits)} poses overlap the box, first {hits[0]}'

    # every command inside the caps, and nothing else publishes /cmd_vel
    assert cmds, 'no /cmd_vel'
    tol = 1e-6
    for vx, vy, vz, wx, wy, wz in cmds:
        assert CAPS['vx'][0] - tol <= vx <= CAPS['vx'][1] + tol, vx
        assert CAPS['vy'][0] - tol <= vy <= CAPS['vy'][1] + tol, vy
        assert CAPS['wz'][0] - tol <= wz <= CAPS['wz'][1] + tol, wz
        assert abs(vz) < tol and abs(wx) < tol and abs(wy) < tol
    assert publishers == ['velocity_smoother'], publishers

    # the map holds the room and nothing else (map and odom both start at the
    # fake's start pose)
    assert grid is not None, 'no /map'
    occupied = fake_g1.occupied_cells(grid)
    far = fake_g1.room_distance(occupied) > MAP_TOLERANCE
    assert len(occupied) > 100, f'only {len(occupied)} occupied cells'
    assert not far.any(), (f'{far.sum()} of {len(occupied)} occupied map cells are more than '
                           f'{MAP_TOLERANCE} m from the room, e.g. {occupied[far][:5].round(2)}')
