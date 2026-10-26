'''The ROS node inside one process: parameters, config, logging, the command topics'''

import time
from pathlib import Path

import pytest
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

from g1_locomotion.locomotion import WAITING
from g1_locomotion.locomotion_node import LocomotionNode

# the package's policy/ (the containers also mount it at /home/code/g1_locomotion)
POLICY_DIR = Path(__file__).resolve().parents[1] / 'policy'
CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)


class FakeDriver:
    '''Stands in for the node's LocomotionDriver: records what the topics set'''

    def __init__(self):
        self.cmd_vel = []
        self.heights = []

    def set_cmd_vel(self, vx, vy, yaw):
        self.cmd_vel.append((vx, vy, yaw))

    def set_height(self, height):
        self.heights.append(height)

    def shutdown(self):
        pass


def _node_config(tmp_path, monkeypatch):
    '''g1.yaml on domain 97, on topics of its own'''
    monkeypatch.setenv('ROS_DOMAIN_ID', str(TEST_DOMAIN))
    with open(CONFIG_DIR / 'g1.yaml') as f:
        raw = yaml.safe_load(f)
    raw['network']['domain_id'] = TEST_DOMAIN
    raw['topics'] = {key: name + '/test_locomotion_node' for key, name in raw['topics'].items()}
    config = tmp_path / 'node.yaml'
    with open(config, 'w') as f:
        yaml.safe_dump(raw, f)
    return config


@pytest.fixture
def node(tmp_path, monkeypatch):
    config = _node_config(tmp_path, monkeypatch)
    rclpy.init(args=['--ros-args', '-p', f'policy_dir:={POLICY_DIR}'], domain_id=TEST_DOMAIN)
    node = LocomotionNode(config_name=str(config))
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_node_starts_silent_and_logs_every_severity(node):
    assert node.get_parameter('policy_dir').value == str(POLICY_DIR)
    assert node._driver.phase == WAITING
    # rclpy refuses a change of severity at one call site, which once hid a fault
    for level in ('info', 'warn', 'error', 'info', 'error'):
        node._log(level, f'test {level}')


def test_topics_reach_the_driver(node):
    node._driver.shutdown()
    node._driver = fake = FakeDriver()
    publisher = rclpy.create_node('test_locomotion_publisher')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(publisher)
    try:
        cmd_vel = publisher.create_publisher(Twist, '/cmd_vel', 10)
        height = publisher.create_publisher(Float32, '/locomotion/height', 10)
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = 0.3, -0.1, 0.2
        twist.linear.z = twist.angular.x = twist.angular.y = 9.0  # not used
        deadline = time.monotonic() + 5.0
        while not (fake.cmd_vel and fake.heights) and time.monotonic() < deadline:
            cmd_vel.publish(twist)
            height.publish(Float32(data=0.6))
            executor.spin_once(timeout_sec=0.05)
        assert fake.cmd_vel and fake.cmd_vel[0] == pytest.approx((0.3, -0.1, 0.2))
        assert fake.heights and fake.heights[0] == pytest.approx(0.6)
    finally:
        executor.shutdown()
        publisher.destroy_node()


def test_sim_time_is_ignored(tmp_path, monkeypatch):
    # the sim bringup passes use_sim_time:=true; the node keeps the monotonic
    # clock and so has no /clock subscription (500 Hz in the sim)
    config = _node_config(tmp_path, monkeypatch)
    rclpy.init(args=['--ros-args', '-p', f'policy_dir:={POLICY_DIR}', '-p', 'use_sim_time:=true'],
               domain_id=TEST_DOMAIN)
    node = LocomotionNode(config_name=str(config))
    try:
        assert node.get_parameter('use_sim_time').value is False
        deadline = time.monotonic() + 5.0
        topics = []
        while '/cmd_vel' not in topics and time.monotonic() < deadline:
            topics = [name for name, _ in
                      node.get_subscriber_names_and_types_by_node('locomotion_node', '/')]
            time.sleep(0.05)
        assert '/cmd_vel' in topics and '/locomotion/height' in topics
        assert '/clock' not in topics
    finally:
        node.destroy_node()
        rclpy.shutdown()
