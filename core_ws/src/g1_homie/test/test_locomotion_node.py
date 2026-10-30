'''
The ROS node inside one process: parameters, config, logging, the command topics, the
heartbeat, the operator's confirm and the band release
'''

import time
from pathlib import Path

import pytest
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Twist

from g1_homie.locomotion import WAITING
from g1_homie.locomotion_node import LocomotionNode

# the package's policy/ (the containers also mount it at /home/code/g1_homie)
POLICY_DIR = Path(__file__).resolve().parents[1] / 'policy'
CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)


class FakeDriver:
    '''Stands in for the node's LocomotionDriver: records what the topics set'''

    def __init__(self, walking_since=None):
        self.cmd_vel = []
        self.heights = []
        self.beats = []
        self.band_released = 0
        self.walking_since = walking_since

    def on_band_released(self):
        self.band_released += 1

    def set_cmd_vel(self, vx, vy, yaw):
        self.cmd_vel.append((vx, vy, yaw))

    def set_height(self, height):
        self.heights.append(height)

    def on_heartbeat(self, beat):
        self.beats.append(beat)

    def shutdown(self):
        pass


def _node_config(tmp_path, monkeypatch, band_release=False):
    '''g1.yaml on domain 97, on topics of its own; band.release as given'''
    monkeypatch.setenv('ROS_DOMAIN_ID', str(TEST_DOMAIN))
    with open(CONFIG_DIR / 'g1.yaml') as f:
        raw = yaml.safe_load(f)
    raw['network']['domain_id'] = TEST_DOMAIN
    raw['topics'] = {key: name + '/test_locomotion_node' for key, name in raw['topics'].items()}
    raw['band']['release'] = band_release
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


def _graph(node):
    '''The node's own (subscriptions, services, clients), by topic or service name'''
    names = (node.get_subscriber_names_and_types_by_node,
             node.get_service_names_and_types_by_node,
             node.get_client_names_and_types_by_node)
    return [[name for name, _ in get('locomotion_node', '/')] for get in names]


def test_node_starts_silent_and_logs_every_severity(node):
    assert node.get_parameter('policy_dir').value == str(POLICY_DIR)
    assert node._driver.phase == WAITING
    # rclpy refuses a change of severity at one call site, which once hid a fault
    for level in ('info', 'warn', 'error', 'info', 'error'):
        node._log(level, f'test {level}')
    # g1.yaml: the heartbeat and the confirm, and no band (no /left_ee_pose, no toggle client)
    deadline = time.monotonic() + 5.0
    subscriptions, services, clients = _graph(node)
    while not ('/safety/heartbeat' in subscriptions and '/lowerbody/confirm_engage' in services):
        assert time.monotonic() < deadline, (subscriptions, services)
        time.sleep(0.05)
        subscriptions, services, clients = _graph(node)
    assert '/left_ee_pose' not in subscriptions and '/elastic_band/toggle' not in clients


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
        heartbeat = publisher.create_publisher(Bool, '/safety/heartbeat', 10)
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = 0.3, -0.1, 0.2
        twist.linear.z = twist.angular.x = twist.angular.y = 9.0  # not used
        deadline = time.monotonic() + 5.0
        while (not (fake.cmd_vel and fake.heights and fake.beats) and
               time.monotonic() < deadline):
            cmd_vel.publish(twist)
            height.publish(Float32(data=0.6))
            heartbeat.publish(Bool(data=True))
            executor.spin_once(timeout_sec=0.05)
        assert fake.cmd_vel and fake.cmd_vel[0] == pytest.approx((0.3, -0.1, 0.2))
        assert fake.heights and fake.heights[0] == pytest.approx(0.6)
        assert fake.beats and set(fake.beats) == {True}
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


def _spin(executor, seconds, done=lambda: False):
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    return done()


def test_confirm_service_and_band_client(tmp_path, monkeypatch):
    # g1.yaml with band.release on, as sim.yaml has it
    config = _node_config(tmp_path, monkeypatch, band_release=True)
    rclpy.init(args=['--ros-args', '-p', f'policy_dir:={POLICY_DIR}'], domain_id=TEST_DOMAIN)
    node = LocomotionNode(config_name=str(config))
    helper = rclpy.create_node('test_locomotion_helper')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    toggles = []

    def toggle(request, response):
        # the band was on: the call takes it off
        toggles.append(time.monotonic())
        response.success = True
        response.message = 'disabled'
        return response

    try:
        helper.create_service(Trigger, '/elastic_band/toggle', toggle)
        confirm = helper.create_client(Trigger, '/lowerbody/confirm_engage')
        pose = helper.create_publisher(PoseStamped, '/left_ee_pose', 10)

        # the operator's confirm, refused while the node waits to start
        assert _spin(executor, 5.0, confirm.service_is_ready)
        future = confirm.call_async(Trigger.Request())
        assert _spin(executor, 5.0, future.done)
        assert not future.result().success and 'waiting' in future.result().message

        # the policy runs: no toggle without /left_ee_pose (band.max_wait is 30 s)
        node._driver.shutdown()
        node._driver = FakeDriver(walking_since=time.monotonic())
        assert not _spin(executor, 0.5, lambda: toggles)
        # one /left_ee_pose: frame_task_server is up, and the band is released, once; the
        # policy's history restarts once the band is off
        assert _spin(executor, 5.0, lambda: pose.get_subscription_count() > 0)
        pose.publish(PoseStamped())
        assert _spin(executor, 5.0, lambda: toggles)
        _spin(executor, 1.0)
        assert len(toggles) == 1, toggles
        assert node._driver.band_released == 1
    finally:
        executor.shutdown()
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def test_band_toggle_that_hangs_the_robot_again_warns_once(tmp_path, monkeypatch):
    # the band was already off (SPACE in the viewer): the toggle hangs the robot again and
    # answers 'enabled'. The node warns, keeps the policy's history and never calls again
    config = _node_config(tmp_path, monkeypatch, band_release=True)
    rclpy.init(args=['--ros-args', '-p', f'policy_dir:={POLICY_DIR}'], domain_id=TEST_DOMAIN)
    node = LocomotionNode(config_name=str(config))
    helper = rclpy.create_node('test_locomotion_helper')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    toggles = []
    log = []
    node._log = lambda level, message: log.append((level, message))

    def toggle(request, response):
        toggles.append(time.monotonic())
        response.success = True
        response.message = 'enabled'
        return response

    try:
        helper.create_service(Trigger, '/elastic_band/toggle', toggle)
        pose = helper.create_publisher(PoseStamped, '/left_ee_pose', 10)
        node._driver.shutdown()
        node._driver = FakeDriver(walking_since=time.monotonic())
        assert _spin(executor, 5.0, node._band_client.service_is_ready)
        assert _spin(executor, 5.0, lambda: pose.get_subscription_count() > 0)
        pose.publish(PoseStamped())
        assert _spin(executor, 5.0, lambda: toggles)
        _spin(executor, 1.0)
        assert len(toggles) == 1, toggles
        assert node._driver.band_released == 0
        warnings = [message for level, message in log if level == 'warn' and '"enabled"' in message]
        assert len(warnings) == 1 and 'is now ON' in warnings[0], log
        assert not any(message.startswith('elastic band released') for _, message in log), log
    finally:
        executor.shutdown()
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
